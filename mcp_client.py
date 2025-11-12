###########################################################################################################################

####################################################################################################################################



import os
import sys
import subprocess
import json
import logging
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client")

# Check for Anthropic API key
if not os.getenv('ANTHROPIC_API_KEY'):
    logger.error("❌ ANTHROPIC_API_KEY not set. MCP requires Claude for SQL generation.")
    raise RuntimeError("ANTHROPIC_API_KEY is required for MCP functionality")

try:
    import anthropic
except ImportError:
    logger.error("❌ anthropic package not installed. Install with: pip install anthropic")
    raise RuntimeError("anthropic package is required")

def patch_sql_for_pagination(sqlquery, offset):
    offset = offset if isinstance(offset, int) and offset >= 0 else 0
    sql_upper = sqlquery.upper()
    if "LIMIT" in sql_upper:
        # Replace existing OFFSET or append if missing
        if "OFFSET" in sql_upper:
            sqlquery = re.sub(r'OFFSET\s+\d+', f'OFFSET {offset}', sqlquery, flags=re.IGNORECASE)
        else:
            sqlquery += f" OFFSET {offset}"
    else:
        sqlquery += f" LIMIT 10 OFFSET {offset}"
    return sqlquery


class MCPPostgreSQLClient:
    def __init__(self):
        # Anthropic is REQUIRED for MCP
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        self.proc = None
        self.schema_cache = None
        
        try:
            self.start_server()
        except Exception as e:
            logger.error(f"❌ MCP Server failed to start: {e}")
            raise RuntimeError(f"Cannot operate without MCP server: {e}")

    def start_server(self):
        """Start MCP PostgreSQL server as subprocess"""
        if self.proc is not None:
            return
            
        logger.info("🚀 Starting MCP PostgreSQL server...")
        
        server_script = "mcp_postgres_server.py"
        if not os.path.exists(server_script):
            raise RuntimeError(f"Server script {server_script} not found")
        
        self.proc = subprocess.Popen(
            [sys.executable, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < 5:
            if self.proc.poll() is not None:
                stderr = self.proc.stderr.read()
                raise RuntimeError(f"MCP server failed: {stderr}")
            
            try:
                line = self.proc.stdout.readline()
                if "READY" in line or "started" in line.lower():
                    logger.info("✅ MCP server started successfully")
                    return
            except:
                pass
            
            time.sleep(0.1)
        
        if self.proc.poll() is None:
            logger.info("✅ MCP server started")
            return
        else:
            raise RuntimeError("MCP server failed to start")

    def send_tool_call(self, tool_name: str, args: dict) -> dict:
        """Send tool call to MCP server"""
        if not self.proc:
            raise RuntimeError("MCP server not running")
        
        req = json.dumps({
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": args
        })
        
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        
        start_time = time.time()
        while time.time() - start_time < 10:
            line = self.proc.stdout.readline()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            time.sleep(0.01)
        
        raise RuntimeError("Timeout waiting for server response")

    def get_database_schema(self) -> Dict[str, Any]:
        """Get database schema from MCP server"""
        if self.schema_cache:
            return self.schema_cache
        
        response = self.send_tool_call("get_database_schema", {})
        if response.get('success'):
            self.schema_cache = response.get('data', {})
            logger.info(f"📊 Schema loaded: {len(self.schema_cache)} tables")
        else:
            logger.error(f"Failed to load schema: {response.get('error')}")
            
        return self.schema_cache or {}

    def natural_language_to_sql(self, user_query: str, org_id=None, client_id=None) -> Dict[str, Any]:
        """
        PURE MCP: Use Claude to dynamically generate SQL based on schema.
        NO HARDCODED PATTERNS!
        """
        tenant_rule = ""
        if org_id and client_id:
            tenant_rule = (
                 f"\n40. ALWAYS add a WHERE clause filtering by BOTH org_id = '{org_id}' AND client_id = '{client_id}'. "
                "Assume every relevant table has org_id and client_id columns. Apply BOTH filters to all queries, even if not explicitly asked. "
                "Do not return data for any other organization or client under any circumstances."
            )
        elif org_id:
            tenant_rule = (
                f"\n40. ALWAYS add a WHERE clause filtering data to ONLY rows with org_id = '{org_id}'. " 
                "Apply this filter to all queries, even if not explicitly asked."
            )
        else:
            tenant_rule = ""

        try:
            # Get fresh schema from MCP server
            schema = self.get_database_schema()
            
            if not schema:
                return {
                    'success': False,
                    'error': 'Cannot generate SQL without database schema',
                    'sql_query': ''
                }
            
            # Build schema description for Claude
            schema_description = self._build_schema_description(schema)
            
            # Create prompt for Claude with full context
            prompt = f"""You are a PostgreSQL expert. Convert the following natural language query to SQL.

                            DATABASE SCHEMA:
                            {schema_description}

                            USER QUERY: {user_query}

                            IMPORTANT RULES:
                            1. ONLY generate SELECT statements (read-only queries)
                            2. Use proper table joins when multiple tables are needed
                            3. Include appropriate WHERE clauses for filtering
                            4. Use LOWER() for case-insensitive text searches
                            5. Always include ORDER BY for consistent results
                            6. Limit results appropriately (default 100 unless specified)
                            7. Use table aliases for clarity (pm for plan_master, pr for plan_rules, etc.)
                            8. For "active" items, check status=1 AND current date between valid_from and valid_to
                            9. Include meaningful column aliases in SELECT
                            10. When asked to "show all" records, do NOT filter by status or date.
                            11. When asked for "active", use status=1 AND current date between valid_from and valid_to
                            12. When in doubt, be literal and avoid filtering unless explicitly requested.
                            13. For "all plans", return all rows in plan_master without filtering by status or dates.
                            14. For “active”, “current”, or “valid”, use the appropriate status=1 AND NOW() BETWEEN valid_from AND valid_to.
                            15. For “inactive”, "expired", "past" etc, use status != 1 OR NOW() NOT BETWEEN valid_from AND valid_to.
                            16. Do NOT add conditions to the SQL if not stated in the question.
                            17. Use COUNT(*), SUM(field), etc, as appropriate.
                            18. Use ORDER BY <date_column> DESC for latest, and ASC for oldest, with LIMIT 1 unless user asks for more.
                            19. Default to LIMIT 10 unless user requests more or all.
                            20. Select only the columns relevant to the question, not all columns, unless user says "all columns" or "full details".
                            21. Always use LOWER(column) = LOWER(value) or ILIKE, especially for textual fields, for robustness.
                            22. When a question involves info from multiple tables (e.g., plan + assignee), join tables using foreign keys.
                            23. If the user query is ambiguous (e.g., "show plans"), default to showing active plans, but state this logic.
                            24. If the user asks “which plans have no <field>” or “missing <field>”, generate WHERE <field> IS NULL or IS NOT NULL as appropriate.
                            25. If query includes "order by", "sort by", follow requested order.
                            26. When returning all plans, include a computed column CASE WHEN status=1 ... as "Active"/"Inactive" for clarity (to allow frontend to display this nicely).
                            27. Never generate UPDATE, INSERT, DELETE, CREATE, ALTER, DROP, TRUNCATE, or any statement other than SELECT.
                            28. Never allow SQL with multiple statements, comments, or batch execution.
                            29. Use ILIKE or LIKE for "contains", "starts with", "ends with" search cases.
                            30. For "missing" values, use IS NULL or IS NULL OR = '' as required.
                            31. For range questions (e.g., "between dates"), use WHERE <date_column> BETWEEN <start> AND <end>.
                            32. For friendly aggregation/group by, use only for count, sum, avg, as requested and group as appropriate.
                            33. Never expose sensitive fields (e.g., passwords, tokens) if they exist.
                            34. Cast date types appropriately when comparing dates and timestamps.
                            35. Always escape and parameterize string literals (never interpolate directly).
                            36. Only reference tables and columns present in the provided schema.
                            37. Limit output using LIMIT and always show the limit.
                            38. If "all columns", expand SELECT to list all column names explicitly.
                            39. If query seems unsupported, dangerous, or ambiguous, return a SELECT that results in zero rows or a warning comment.
                            40. {tenant_rule}

                            Return ONLY the SQL query, no explanation or markdown."""

            # Claude generates SQL dynamically based on YOUR schema
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0,  # Deterministic output
                messages=[{"role": "user", "content": prompt}]
            )
            
            sql_query = response.content[0].text.strip()
            
            # Clean up any markdown or code blocks Claude might add
            sql_query = sql_query.replace('```sql', '').replace('```', '')
            sql_query = sql_query.strip().rstrip(';')
            
            # Log the generated query
            logger.info("="*60)
            logger.info(f"🔍 USER QUERY: {user_query}")
            logger.info(f"🤖 CLAUDE GENERATED SQL: {sql_query}")
            logger.info("="*60)
            
            return {
                'success': True,
                'sql_query': sql_query,
                'original_query': user_query,
                'method': 'claude_dynamic'  # Track that this was dynamically generated
            }
            
        except Exception as e:
            logger.error(f"❌ SQL generation failed: {e}")
            err_txt = str(e)
            if "429" in err_txt or "rate_limit_error" in err_txt or "Too Many Requests" in err_txt:
                return {
                    'success': False,
                    'error': 'The system is currently handling too many requests to the AI service. Please wait a few seconds and try again.', #f'Cannot generate SQL: {str(e)}',
                    'sql_query': '',
                    'original_query': user_query
                }
            return {
                'success': False,
                'error': 'Sorry, an internal error occurred while generating SQL.',
                'sql_query': '',
                'original_query': user_query
            }

    def _build_schema_description(self, schema: dict) -> str:
        """Build detailed schema description for Claude"""
        lines = []
        
        # Add overview
        lines.append(f"Database contains {len(schema)} tables.\n")
        
        # Add detailed table information
        for table_name, table_info in schema.items():
            lines.append(f"\nTable: {table_name}")
            lines.append(f"Purpose: {table_info.get('description', 'Data table')}")
            lines.append("Columns:")
            
            for col in table_info.get('columns', []):
                col_desc = f"  - {col['name']} ({col['type']})"
                
                # Add constraints
                constraints = []
                if col.get('primary_key'):
                    constraints.append("PRIMARY KEY")
                if not col.get('nullable', True):
                    constraints.append("NOT NULL")
                if col.get('foreign_key'):
                    fk = col['foreign_key']
                    constraints.append(f"FK→{fk['table']}.{fk['column']}")
                
                if constraints:
                    col_desc += f" [{', '.join(constraints)}]"
                
                lines.append(col_desc)
            
            # Add relationships if any
            if table_info.get('relationships'):
                lines.append("  Relationships:")
                for rel in table_info['relationships']:
                    lines.append(f"    - {rel}")
        
        return "\n".join(lines)

    def execute_query_with_claude(self, user_query: str, org_id=None, client_id=None, offset=0) -> Dict[str, Any]:
        """Complete MCP pipeline: NL → SQL → Execute → Format"""
        try:
            # Step 1: Generate SQL using Claude (DYNAMIC)
            
            sql_result = self.natural_language_to_sql(user_query, org_id=org_id, client_id=client_id)
            sql_result['sql_query'] = patch_sql_for_pagination(sql_result['sql_query'], offset)
            '''# BEGIN PATCH
            offset = getattr(self, "plan_offset", 0) if hasattr(self, "plan_offset") else 0
            sqlquery = sql_result['sql_query']
            if "LIMIT" in sqlquery.upper():
                if "OFFSET" not in sqlquery.upper():
                    sqlquery += f" OFFSET {offset}"
            else:
                sqlquery += f" LIMIT 10 OFFSET {offset}"
            sql_result['sql_query'] = sqlquery
            # END PATCH'''

            exec_result = self.send_tool_call("execute_sql_query", {
            "sql": sql_result['sql_query']
            })

            if not sql_result.get('success'):
                return {
                    'success': False,
                    'error': sql_result.get('error', 'Failed to generate SQL'),
                    'formatted_response': f"I couldn't understand your query: {sql_result.get('error', 'Unknown error')}"
                }
            
            # Step 2: Execute SQL via MCP server
            logger.info(f"📊 EXECUTING SQL via MCP Server")
            exec_result = self.send_tool_call("execute_sql_query", {
                "sql": sql_result['sql_query']
            
            })
            
            if exec_result.get('success'):
                logger.info(f"✅ QUERY SUCCESSFUL: {exec_result.get('row_count', 0)} rows returned")
            else:
                logger.error(f"❌ QUERY FAILED: {exec_result.get('error')}")
                return {
                    'success': False,
                    'error': exec_result.get('error'),
                    'formatted_response': f"Database query failed: {exec_result.get('error')}"
                }
            
            # Step 3: Format results in plain English using Claude
            formatted = self.format_results_as_english(
                exec_result.get('data', []),
                user_query
            )
            
            return {
                'success': True,
                'data': exec_result.get('data', []),
                "response": formatted,
                'formatted_response': formatted,
                'sql_query': sql_result['sql_query'],
                'row_count': exec_result.get('row_count', 0),
                'method': sql_result.get('method', 'mcp')
            }
            
        except Exception as e:
            logger.error(f"❌ MCP pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'formatted_response': f"Sorry, I encountered an error processing your query: {str(e)}"
            }

    def format_results_as_english(self, data: list, original_query: str) -> str:
        """Use Claude to format database results as natural English"""
        if not data:
            return "I couldn't find any data matching your query."
        
        try:
            # Limit data for context window
            sample_data = data[:20] if len(data) > 20 else data
            data_json = json.dumps(sample_data, default=str)
            
            prompt = f"""Convert this database query result into natural, conversational English.

                            Original Question: {original_query}

                            Database Results ({len(data)} total rows):
                            {data_json}

                            Instructions:
                            1. For every row/result, output a new bullet point that is always a FULL standalone English sentence, not just a list of fields, keywords, or a fragment. Each bullet must directly state all the relevant fields in a smooth, natural, business style (e.g., "• The 'testghj' plan is valid from Sep 8, 2025 to Sep 9, 2026 and is currently pending approval and review.").
                            2. Do not use shorthand, telegraphic style, or field lists. Each bullet must be a sentence that can stand alone when read aloud.
                            3. Present each row/result as a separate bullet point (•) or numbered list item, regardless of what type of entity or table is being shown.
                            4. For each result, include all relevant human-readable fields (such as names, dates, status, type, etc.) that would be helpful for a business user. If fields are not relevant to the user, omit them. If the entity type is clear (e.g., program, plan, employee), start the bullet with that name.
                            5. Do not group, summarize, or collapse items—there must be exactly one bullet/list entry per row in the results, even for a large result set.
                            6. If showing more than 20 results, only show the first 20 (one bullet each), and clearly state at the end how many more exist.
                            7. Avoid using SQL/database or technical terminology. Write for a non-technical business audience.
                            8. Be concise but complete: prefer short, direct, clear sentences, but do not leave out key business details.
                            9. If the result is for a count, statistic, or aggregation rather than rows, state the answer in one clear sentence.
                            10. Never return table/field names or JSON unless they are required for clarity.
                            11. The output must be easy to read and scan, and friendly for a business user. Never say "here is the query result:"
                            12. Always mention the total count of results if not all are shown.
                            13. Never mention that you are an AI, a language model, or anything about SQL or technical processes.
                            14. For each result/row, output a new line that starts with a Markdown bullet (`*`) or Unicode bullet (`•`), followed by that result's info—never as a numbered inline list, but with each bullet/result on its own line.
                            15. Include key details for each row as appropriate for business users (plan name, date, type, status, etc).
                            16. Never mention SQL, queries, or technical terms.
                            17. Format so it is easily readable in plain text or Markdown—use hard line breaks between bullets!
                            18. If the result is just a single value, return a clear sentence.
                            
                            Response:"""


            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            formatted_response = response.content[0].text.strip()
            
            # Add total count if there are more rows than shown
            if len(data) > 20:
                formatted_response += f"\n\n(Showing highlights from {len(data)} total results)"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to format with Claude: {e}")
            # Last resort fallback - just show row count
            return f"Found {len(data)} results matching your query."

    def process_natural_language_query(self, user_query: str, context: str = "", org_id=None, client_id=None, offset=0) -> Dict[str, Any]:
        """Main entry point for MCP query processing"""
        logger.info(f"🎯 Processing query: '{user_query}' (org_id={org_id}, client_id={client_id})")
        return self.execute_query_with_claude(user_query, org_id=org_id, client_id=client_id, offset=offset)
    
    def claude_complete(self, prompt, max_tokens=400, temperature=0):
        """
        Call Claude (Anthropic) directly for generic completions with plain text output.
        """
        try:
            if not self.anthropic_client:
                raise RuntimeError("Anthropic client not initialized")
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit_error" in str(e) or "Too Many Requests" in str(e):
                return "The server is handling too many requests at the moment. Please try again in a few seconds."
            raise

    def __del__(self):
        """Cleanup MCP server on exit"""
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except:
                pass

# Initialize MCP client
try:
    mcp_client = MCPPostgreSQLClient()
    logger.info("✅ MCP Client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize MCP client: {e}")
    raise
