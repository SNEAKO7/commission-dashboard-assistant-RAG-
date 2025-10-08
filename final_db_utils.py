import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Enhanced wizard options with more comprehensive mappings
WIZARD_OPTIONS = {
    "category type": ["Flat", "Slab", "Tiered"],
    "range type": ["Amount", "Quantity"],
    "base value": ["Gross", "Net"],
    "plan base": ["Revenue", "Margin"],
    "value type": ["Percentage", "Amount"],
    "object type": ["Invoices", "Contracts", "Sales Orders"],
    "plan parameters": ["Territory", "Product", "BusinessPartner", "Plant", "Group"],
    "status": ["Active", "Inactive"],
    "frequency": ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
}

def _inject_tenant_filters(sql, tenant_filters):
    """
    Adds WHERE ...field1 = %s AND field2 = %s... clause to SELECTs on known tables.
    tenant_filters: list of (field_name, value)
    """
    if not sql.strip().lower().startswith("select"):
        return sql, ()
    s = sql.lower()
    for tab in ["plan_master", "plan_rules", "commission_run", "commission_run_details", "plan_assignment", "plan_value_ranges"]:
        if f" from {tab}" in s or f"join {tab}" in s:
            # List of conditions
            clause = " AND ".join([f"{field} = %s" for field, value in tenant_filters if value is not None])
            values = tuple([value for field, value in tenant_filters if value is not None])
            if " where " in s:
                idx = s.index(" where ") + 7
                return sql[:idx] + clause + " AND " + sql[idx:], values
            else:
                # before ORDER BY/LIMIT if present, else at end
                parts = re.split(r"(order by|limit)", sql, flags=re.IGNORECASE)
                if len(parts) > 1:
                    return parts[0] + f" WHERE {clause} " + ''.join(parts[1:]), values
                else:
                    return sql + f" WHERE {clause}", values
    return sql, ()


def smart_wizard_option_answer(q):
    """Enhanced wizard option answering with better matching"""
    lowerq = q.strip().lower()
    
    # Check for specific option requests
    for field, opts in WIZARD_OPTIONS.items():
        if any(phrase in lowerq for phrase in [
            f"what {field}", f"{field} options", f"available {field}", 
            f"possible {field}", f"valid {field}", f"list {field}",
            f"show {field}", f"all {field}"
        ]):
            return f"The available options for {field} are: {', '.join(opts)}"
    
    # Check for individual option validation
    for field, opts in WIZARD_OPTIONS.items():
        for opt in opts:
            if f"is {opt.lower()} valid" in lowerq or f"can i use {opt.lower()}" in lowerq:
                return f"Yes, '{opt}' is a valid {field} option."
    
    return None

class DatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'polaris.callippus.co.uk'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'commissions'),
            'user': os.getenv('DB_USER', 'commissionsuser'),
            'password': os.getenv('DB_PASSWORD', 'Commissionsuser@123'),
        }
        self._schema_cache = None
        self._connection = None

    def get_connection(self):
        """Enhanced connection with proper transaction handling"""
        try:
            if self._connection is None or self._connection.closed:
                self._connection = psycopg2.connect(**self.connection_params)
                self._connection.autocommit = True  # Enable autocommit to avoid transaction issues
            return self._connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Reset connection on failure
            self._connection = None
            raise

    '''def get_schema(self) -> Dict[str, Any]:
        """Enhanced schema loading with actual column checking"""
        if self._schema_cache is not None:
            return self._schema_cache

        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all relevant tables with UPDATED names based on your new schema
                cur.execute("""
                SELECT t.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default
                FROM information_schema.tables t
                JOIN information_schema.columns c ON t.table_name = c.table_name
                WHERE t.table_schema = 'public'
                AND t.table_type = 'BASE TABLE'
                AND t.table_name IN (
                    'plan_master', 'plan_rules', 'plan_assignment',
                    'plan_value_ranges', 'commission_schedule', 'plan_type_master',
                    'commission_run', 'commission_run_details'
                )
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

                self._schema_cache = schema
                logger.info(f"Loaded schema for {len(schema)} tables")
                return schema

        except Exception as e:
            logger.error(f"Failed to load database schema: {e}")
            return {}'''

    def get_schema(self) -> Dict[str, Any]:
        #Loads full schema info for *all* tables in 'public' schema from PostgreSQL. Returns a dict: {table_name: {columns: [{name, type, nullable, default}], description: str}}"""
        if self._schema_cache is not None:
            return self._schema_cache

        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get ALL BASE TABLES (remove any hardcoded IN (...))
                cur.execute("""
                    SELECT t.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default
                    FROM information_schema.tables t
                    JOIN information_schema.columns c ON t.table_name = c.table_name
                    WHERE t.table_schema = 'public'
                    AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name, c.ordinal_position
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
                self._schema_cache = schema
            return schema
        except Exception as e:
            logger.error(f"Failed to load database schema: {e}")
            return {}


    def _get_table_description(self, table_name: str) -> str:
        """Enhanced table descriptions with UPDATED table names"""
        descriptions = {
            'plan_master': 'Master table for programs/plans with basic program information',
            'plan_rules': 'Detailed rules, conditions, types, and logic of plans',
            'plan_assignment': 'Assignment/configuration of plan parameters and values',
            'plan_type_master': 'Master list of all available plan types',
            'plan_value_ranges': 'Value ranges, tiers, and commission rates for calculations',
            'commission_schedule': 'Schedule metadata (scheduler_name, frequency, timing)',
            'commission_run': 'Commission calculation run records and results',
            'commission_run_details': 'Detailed records of each commission run execution'
        }
        return descriptions.get(table_name, f'Table: {table_name}')

    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Enhanced SQL validation"""
        if not sql:
            return False, "Empty SQL query"
        
        sql_lower = sql.lower().strip()
        
        # Only allow SELECT statements
        if not sql_lower.startswith('select'):
            return False, "Only SELECT queries are allowed"
        
        # Enhanced dangerous keyword detection
        dangerous_patterns = [
            r'\binsert\b', r'\bupdate\b', r'\bdelete\b', r'\bdrop\b',
            r'\bcreate\b', r'\balter\b', r'\btruncate\b', r'\bexec\b', 
            r'\bexecute\b', r'\bsp_\b', r'\bxp_\b', r'\bgrant\b', 
            r'\brevoke\b', r'\binto\s+outfile\b', r'\bload_file\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_lower):
                return False, f"Dangerous operation '{pattern}' not allowed"
        
        # Check for potential SQL injection patterns
        injection_patterns = [
            r';--', r'/\*', r'\*/', r'\bunion\s+select\b', 
            r'\bor\s+1\s*=\s*1\b', r'\bor\s+true\b', r'--\s*$',
            r'\bconcat\s*\(', r'\bsubstring\s*\(', r'\bascii\s*\('
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql_lower):
                return False, f"Potential SQL injection pattern detected: {pattern}"
        
        return True, "Query validated"

    def execute_safe_query(self, sql: str, params: Optional[tuple] = None, tenant_id=None, client_id=None, tenant_field='org_id', client_field='client_id') -> Dict[str, Any]:
        """Enhanced query execution with better error handling and transaction management"""
        logger.info(f"🗄️ [DB] Executing SQL: {sql}")
        if params:
            logger.info(f"🗄️ [DB] Parameters: {params}")
        
        # Validate SQL first
        is_valid, validation_msg = self.validate_sql(sql)
        if not is_valid:
            return {
                'success': False,
                'error': validation_msg,
                'data': [],
                'row_count': 0
            }
        # Inject tenant filter if tenant_id is provided
        if tenant_id is not None:
            orig_params = params or ()
            tenant_filters = []
        if tenant_id is not None:
                tenant_filters.append((tenant_field, tenant_id))
        if client_id is not None:
                tenant_filters.append((client_field, client_id))

        if tenant_filters:
                sql, filter_values = _inject_tenant_filters(sql, tenant_filters)
                if filter_values:
                    params = filter_values + (orig_params if orig_params else ())
                else:
                    params = orig_params


        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                
                # Convert to regular dicts and handle datetime objects
                data = []
                for row in rows:
                    row_dict = dict(row)
                    for k, v in row_dict.items():
                        if isinstance(v, datetime):
                            row_dict[k] = v.isoformat()
                    data.append(row_dict)

                logger.info(f"🗄️ [DB] Query successful. {len(data)} row(s) returned.")
                
                return {
                    'success': True,
                    'data': data,
                    'row_count': len(data),
                    'executed_sql': sql,
                    'params': params
                }

        except Exception as e:
            logger.error(f"🗄️ [DB] Query execution failed: {e}")
            # Reset connection on error to avoid transaction issues
            try:
                if self._connection:
                    self._connection.close()
            except:
                pass
            self._connection = None
            
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'row_count': 0,
                'executed_sql': sql
            }

    def build_sql_from_nl(self, question: str, schema: Dict[str, Any]) -> Tuple[Optional[str], Optional[tuple]]:
        """Deprecated - SQL generation now handled by Claude via MCP"""
        logger.warning("build_sql_from_nl called directly - should use MCP client instead")
        return None, None

# Create global instance
db_manager = DatabaseManager()
