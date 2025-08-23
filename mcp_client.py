#!/usr/bin/env python3
"""
MCP Client - Pure MCP implementation without hardcoded SQL
"""

import os
import sys
import subprocess
import json
import logging
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv

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

    def natural_language_to_sql(self, user_query: str) -> Dict[str, Any]:
        """
        PURE MCP: Use Claude to dynamically generate SQL based on schema.
        NO HARDCODED PATTERNS!
        """
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

Return ONLY the SQL query, no explanation or markdown."""

            # Claude generates SQL dynamically based on YOUR schema
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
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
            return {
                'success': False,
                'error': f'Cannot generate SQL: {str(e)}',
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

    def execute_query_with_claude(self, user_query: str) -> Dict[str, Any]:
        """Complete MCP pipeline: NL → SQL → Execute → Format"""
        try:
            # Step 1: Generate SQL using Claude (DYNAMIC)
            sql_result = self.natural_language_to_sql(user_query)
            
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
1. Provide a natural, conversational response
2. Don't mention SQL, queries, databases, or technical terms
3. If there are many results, summarize the key information
4. Use bullet points or numbering for lists when appropriate
5. Be concise but complete
6. If showing a list, include the most important/relevant items
7. Mention the total count if there are many results

Response:"""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
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

    def process_natural_language_query(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """Main entry point for MCP query processing"""
        logger.info(f"🎯 Processing query: '{user_query}'")
        return self.execute_query_with_claude(user_query)

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
