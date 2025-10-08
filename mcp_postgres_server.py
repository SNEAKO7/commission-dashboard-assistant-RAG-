#!/usr/bin/env python3
"""
PostgreSQL MCP Server - Provides database tools
"""

import sys
import json
import logging
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError as e:
    print(f"Error: psycopg2 not installed. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    from sql_security import validate_sql_security
except ImportError:
    # Fallback if sql_security.py doesn't exist
    def validate_sql_security(sql):
        sql = sql.strip()
        if sql.upper().startswith('SELECT'):
            return True, "OK", sql
        return False, "Only SELECT allowed", None

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_postgres_server")

class MCPPostgreSQLServer:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'polaris.callippus.co.uk'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'commissions'),
            'user': os.getenv('DB_USER', ''),
            'password': os.getenv('DB_PASSWORD', ''),
        }
        self._connection = None
        logger.info(f"MCP Server initialized for {self.connection_params['host']}:{self.connection_params['port']}")

    def get_connection(self):
        try:
            if self._connection is None or self._connection.closed:
                self._connection = psycopg2.connect(**self.connection_params)
                self._connection.autocommit = True
            return self._connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def get_schema(self):
        """Get complete database schema"""
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        t.table_name, 
                        c.column_name, 
                        c.data_type, 
                        c.is_nullable,
                        c.column_default
                    FROM information_schema.tables t
                    JOIN information_schema.columns c ON t.table_name = c.table_name
                    WHERE t.table_schema = 'public' 
                    AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name, c.ordinal_position;
                """)
                
                schema = {}
                for row in cur.fetchall():
                    table_name = row['table_name']
                    if table_name not in schema:
                        schema[table_name] = {
                            'columns': [],
                            'description': self._get_table_description(table_name)
                        }
                    
                    schema[table_name]['columns'].append({
                        'name': row['column_name'],
                        'type': row['data_type'],
                        'nullable': row['is_nullable'] == 'YES',
                        'default': row['column_default']
                    })
                
                return schema
                
        except Exception as e:
            logger.error(f"Schema loading failed: {e}")
            return {}

    def _get_table_description(self, table_name: str) -> str:
        descriptions = {
            'plan_master': 'Master table containing commission programs',
            'plan_rules': 'Commission plan calculation rules',
            'plan_assignment': 'Plan to assignee mappings',
            'plan_value_ranges': 'Commission tiers and ranges',
            'commission_schedule': 'Payment and calculation schedules',
            'plan_type_master': 'Available plan types',
            'commission_run': 'Commission calculation runs',
            'commission_run_details': 'Detailed commission calculations'
        }
        return descriptions.get(table_name, f'Table: {table_name}')

    def execute_sql(self, sql: str):
        """Execute SQL query with security validation"""
        try:
            is_valid, reason, cleaned_sql = validate_sql_security(sql)
            if not is_valid:
                return {
                    'success': False,
                    'error': f'Security: {reason}',
                    'data': []
                }
            
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(cleaned_sql)
                rows = cur.fetchall()
                
                data = []
                for row in rows:
                    row_dict = dict(row)
                    for k, v in row_dict.items():
                        if isinstance(v, datetime):
                            row_dict[k] = v.isoformat()
                    data.append(row_dict)
                
                return {
                    'success': True,
                    'data': data,
                    'row_count': len(data)
                }
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': []
            }

    def handle_request(self, request):
        """Handle incoming requests"""
        tool_name = request.get('tool_name')
        args = request.get('arguments', {})
        
        if tool_name == 'get_database_schema':
            schema = self.get_schema()
            return {'success': True, 'data': schema}
        
        elif tool_name == 'execute_sql_query':
            sql = args.get('sql', '')
            return self.execute_sql(sql)
        
        else:
            return {'success': False, 'error': f'Unknown tool: {tool_name}'}

def main():
    """Run the MCP server"""
    try:
        server = MCPPostgreSQLServer()
        logger.info("🚀 MCP PostgreSQL Server started successfully")
        print("READY", flush=True)  # Signal that server is ready
        
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            try:
                request = json.loads(line)
                response = server.handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                continue
            except Exception as e:
                error_response = {'success': False, 'error': str(e)}
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
                
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        print(json.dumps({'success': False, 'error': str(e)}), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
