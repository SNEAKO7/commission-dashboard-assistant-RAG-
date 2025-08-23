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
            'host': os.getenv('DB_HOST', ' '),
            'port': int(os.getenv('DB_PORT', ' ')),
            'database': os.getenv('DB_NAME', ' '),
            'user': os.getenv('DB_USER', ' '),
            'password': os.getenv('DB_PASSWORD', ' '),
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

    def execute_safe_query(self, sql: str, params: Optional[tuple] = None) -> Dict[str, Any]:
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
# natural language to SQL conversion with proper pattern matching IF NEEDED Uncomment
    '''def build_sql_from_nl(self, question: str, schema: Dict[str, Any]) -> Tuple[Optional[str], Optional[tuple]]:
        """
        COMPLETELY FIXED: Enhanced natural language to SQL conversion with proper pattern matching
        """
        q = question.lower().strip()
        logger.info(f"[NL2SQL] Processing query: '{question}'")
        
        # First check for wizard options - only when explicitly asking for options
        if any(phrase in q for phrase in ["what are the", "options for", "available options", "possible values", "valid options"]):
            wizard_ans = smart_wizard_option_answer(question)
            if wizard_ans is not None:
                # Return a simple query that will show the options
                return "SELECT 'Options' as info, %s as details;", (wizard_ans,)
        
        # Use plan_master as primary table
        primary_table = "plan_master"
        
        # PRIORITY 1: FIXED Count queries (must be first!)
        if any(word in q for word in ['count', 'how many', 'number of', 'total']):
            if any(word in q for word in ['program', 'programs']):
                if any(word in q for word in ['active', 'current', 'valid']):
                    sql = f"SELECT COUNT(DISTINCT program_name) as active_program_count FROM {primary_table} WHERE status=1 AND NOW() BETWEEN valid_from AND valid_to;"
                    logger.info(f"[NL2SQL] Active program count query: {sql}")
                    return sql, None
                elif any(word in q for word in ['inactive', 'expired']):
                    sql = f"SELECT COUNT(DISTINCT program_name) as inactive_program_count FROM {primary_table} WHERE status!=1 OR NOW() NOT BETWEEN valid_from AND valid_to;"
                    logger.info(f"[NL2SQL] Inactive program count query: {sql}")
                    return sql, None
                else:
                    sql = f"SELECT COUNT(DISTINCT program_name) as total_program_count FROM {primary_table};"
                    logger.info(f"[NL2SQL] Total program count query: {sql}")
                    return sql, None
            
            if any(word in q for word in ['plan', 'plans']):
                sql = f"""
                SELECT COUNT(*) as total_plan_count
                FROM plan_rules pr
                JOIN {primary_table} pm ON pm.id = pr.plan_id
                WHERE pm.status=1;
                """
                logger.info(f"[NL2SQL] Plan count query: {sql}")
                return sql, None
        # PRIORITY 2: FIXED Assignee queries (must be before general show queries!)
        if any(word in q for word in ['assignee', 'assignees']) and any(word in q for word in ['show', 'list', 'all', 'name', 'names']):
            sql = f"""
            SELECT DISTINCT pm.assignee_name, COUNT(*) as program_count
            FROM {primary_table} pm
            WHERE pm.assignee_name IS NOT NULL AND pm.assignee_name != ''
            GROUP BY pm.assignee_name
            ORDER BY pm.assignee_name;
            """
            logger.info(f"[NL2SQL] Assignee names query: {sql}")
            return sql, None
        # PRIORITY 3: FIXED Plan type queries
        if 'plan type' in q or 'plan types' in q:
            if any(word in q for word in ['available', 'all', 'list', 'show', 'what are']):
                if "plan_type_master" in schema:
                    # Check if description column exists
                    plan_type_columns = [col['name'] for col in schema["plan_type_master"]["columns"]]
                    if 'description' in plan_type_columns:
                        sql = "SELECT DISTINCT plan_name as plan_type, description FROM plan_type_master ORDER BY plan_name;"
                    else:
                        sql = "SELECT DISTINCT plan_name as plan_type FROM plan_type_master ORDER BY plan_name;"
                else:
                    sql = "SELECT DISTINCT plan_type, COUNT(*) as usage_count FROM plan_rules WHERE plan_type IS NOT NULL GROUP BY plan_type ORDER BY plan_type;"
                logger.info(f"[NL2SQL] Available plan types query: {sql}")
                return sql, None
        # PRIORITY 4: FIXED Active/Inactive plan queries with proper JOIN
        rules_table = "plan_rules"
        
        if re.search(r'\bplans?\b', q) and 'plan type' not in q:
            # FIXED: Active plans
            if any(word in q for word in ['active', 'current', 'valid']):
                sql = f"""
                SELECT pm.program_name, pr.plan_type, pr.category_type,
                       pr.range_type, pr.value_type
                FROM {primary_table} pm
                JOIN {rules_table} pr ON pm.id = pr.plan_id
                WHERE pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
                ORDER BY pm.program_name, pr.sequence;
                """
                logger.info(f"[NL2SQL] Active plans query: {sql}")
                return sql, None
            
            # FIXED: Inactive plans (DIFFERENT SQL!)
            elif any(word in q for word in ['inactive', 'expired', 'not active']):
                sql = f"""
                SELECT pm.program_name, pr.plan_type, pr.category_type,
                       pr.range_type, pr.value_type
                FROM {primary_table} pm
                JOIN {rules_table} pr ON pm.id = pr.plan_id
                WHERE pm.status!=1 OR NOW() NOT BETWEEN pm.valid_from AND pm.valid_to
                ORDER BY pm.program_name, pr.sequence;
                """
                logger.info(f"[NL2SQL] Inactive plans query: {sql}")
                return sql, None
            
            # All plans or just "show plans"
            else:
                sql = f"""
                SELECT pm.program_name, pr.plan_type, pr.category_type,
                       pr.range_type, pr.value_type,
                       CASE WHEN pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
                            THEN 'Active' ELSE 'Inactive' END as status
                FROM {primary_table} pm
                JOIN {rules_table} pr ON pm.id = pr.plan_id
                ORDER BY pm.program_name, pr.sequence;
                """
                logger.info(f"[NL2SQL] All plans query: {sql}")
                return sql, None
        # PRIORITY 5: FIXED Program queries with object type detection
        program_patterns = [
            r'programs?\s+(?:with|having)\s+(?:object\s+type\s+)?(\w+)',
            r'(?:list|show)\s+programs?\s+(?:with|of|for)\s+(\w+)',
            r'programs?\s+(?:for|with)\s+(\w+)'
        ]
        
        for pattern in program_patterns:
            match = re.search(pattern, q)
            if match:
                criteria = match.group(1).lower()
                
                # Object type queries
                if criteria in ['invoice', 'invoices', 'contract', 'contracts', 'sales', 'order', 'orders']:
                    object_map = {
                        'invoice': 'Invoices', 'invoices': 'Invoices',
                        'contract': 'Contracts', 'contracts': 'Contracts', 
                        'sales': 'Sales Orders', 'order': 'Sales Orders', 'orders': 'Sales Orders'
                    }
                    obj_type = object_map.get(criteria, criteria.title())
                    
                    sql = f"""
                    SELECT DISTINCT pm.program_name, pm.object_type, pm.assignee_name,
                           pm.valid_from, pm.valid_to,
                           CASE WHEN pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
                                THEN 'Active' ELSE 'Inactive' END as status
                    FROM {primary_table} pm
                    WHERE LOWER(pm.object_type) LIKE LOWER(%s)
                    ORDER BY pm.program_name;
                    """
                    logger.info(f"[NL2SQL] Programs with object type '{obj_type}' query: {sql}")
                    return sql, (f"%{obj_type}%",)
        # PRIORITY 6: FIXED Program status queries
        if any(word in q for word in ['program', 'programs']):
            count_requested = 'count' in q or 'how many' in q or 'number' in q
            
            # Active programs
            if any(word in q for word in ['active', 'current', 'running', 'valid']):
                if count_requested:
                    sql = f"SELECT COUNT(DISTINCT program_name) as active_program_count FROM {primary_table} WHERE status=1 AND NOW() BETWEEN valid_from AND valid_to;"
                else:
                    sql = f"""
                    SELECT DISTINCT pm.program_name, pm.assignee_name, pm.object_type,
                           pm.valid_from, pm.valid_to, 'Active' as status
                    FROM {primary_table} pm
                    WHERE pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
                    ORDER BY pm.program_name;
                    """
                logger.info(f"[NL2SQL] Active programs query: {sql}")
                return sql, None
            
            # FIXED: Inactive programs
            elif any(word in q for word in ['inactive', 'expired', 'not active']):
                if count_requested:
                    sql = f"SELECT COUNT(DISTINCT program_name) as inactive_program_count FROM {primary_table} WHERE status!=1 OR NOW() NOT BETWEEN valid_from AND valid_to;"
                else:
                    sql = f"""
                    SELECT DISTINCT pm.program_name, pm.assignee_name, pm.object_type,
                           pm.valid_from, pm.valid_to, 'Inactive' as status
                    FROM {primary_table} pm
                    WHERE pm.status!=1 OR NOW() NOT BETWEEN pm.valid_from AND pm.valid_to
                    ORDER BY pm.program_name;
                    """
                logger.info(f"[NL2SQL] Inactive programs query: {sql}")
                return sql, None
            
            # All programs
            elif any(word in q for word in ['all', 'show', 'list']):
                if count_requested:
                    sql = f"SELECT COUNT(DISTINCT program_name) as total_program_count FROM {primary_table};"
                else:
                    sql = f"""
                    SELECT DISTINCT pm.program_name, pm.assignee_name, pm.object_type,
                           pm.valid_from, pm.valid_to,
                           CASE WHEN pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
                                THEN 'Active' ELSE 'Inactive' END as status
                    FROM {primary_table} pm
                    ORDER BY pm.program_name;
                    """
                logger.info(f"[NL2SQL] All programs query: {sql}")
                return sql, None
        # PRIORITY 7: Schedule queries
        if any(word in q for word in ['schedule', 'schedules']):
            if any(word in q for word in ['all', 'list', 'show', 'available']):
                sql = """
                SELECT id, scheduler_name, frequency, run_time, start_date,
                       CASE WHEN is_active THEN 'Active' ELSE 'Inactive' END as status
                FROM commission_schedule
                WHERE is_active = true
                ORDER BY scheduler_name;
                """
                logger.info(f"[NL2SQL] All schedules query: {sql}")
                return sql, None
        # PRIORITY 8: Specific program details
        if any(word in q for word in ['details', 'info', 'information']):
            program_match = re.search(r'(?:details|info|information)\s+(?:for|of|about)\s+(["\']?)([^"\']+)\1', q)
            if program_match:
                program_name = program_match.group(2).strip()
                sql = f"""
                SELECT pm.program_name, pm.assignee_name, pm.object_type,
                       pm.valid_from, pm.valid_to,
                       cs1.scheduler_name as payment_schedule,
                       cs2.scheduler_name as calculation_schedule,
                       CASE WHEN pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
                            THEN 'Active' ELSE 'Inactive' END as status
                FROM {primary_table} pm
                LEFT JOIN commission_schedule cs1 ON pm.payment_schedule = cs1.id
                LEFT JOIN commission_schedule cs2 ON pm.calculation_schedule = cs2.id
                WHERE LOWER(pm.program_name) LIKE LOWER(%s);
                """
                logger.info(f"[NL2SQL] Program details query: {sql}")
                return sql, (f"%{program_name}%",)
        # PRIORITY 9: Date range queries
        date_match = re.search(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', q)
        if date_match:
            start_date, end_date = date_match.group(1), date_match.group(2)
            sql = f"""
            SELECT DISTINCT pm.program_name, pm.valid_from, pm.valid_to,
                   CASE WHEN pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
                        THEN 'Active' ELSE 'Inactive' END as status,
                   pm.assignee_name, pm.object_type
            FROM {primary_table} pm
            WHERE ((pm.valid_from >= %s AND pm.valid_from <= %s)
                OR (pm.valid_to >= %s AND pm.valid_to <= %s)
                OR (pm.valid_from <= %s AND pm.valid_to >= %s))
            ORDER BY pm.valid_from DESC;
            """
            logger.info(f"[NL2SQL] Date range query: {sql}")
            return sql, (start_date, end_date, start_date, end_date, start_date, end_date)
        # DEFAULT FALLBACK: Show recent active programs with key information
        logger.info(f"[NL2SQL] Using fallback query for: {question}")
        sql = f"""
        SELECT DISTINCT pm.program_name, pm.assignee_name, pm.object_type,
               pm.valid_from, pm.valid_to, 'Active' as status
        FROM {primary_table} pm
        WHERE pm.status=1 AND NOW() BETWEEN pm.valid_from AND pm.valid_to
        ORDER BY pm.program_name
        LIMIT 10;
        """
        return sql, None'''

    def build_sql_from_nl(self, question: str, schema: Dict[str, Any]) -> Tuple[Optional[str], Optional[tuple]]:
        """Deprecated - SQL generation now handled by Claude via MCP"""
        logger.warning("build_sql_from_nl called directly - should use MCP client instead")
        return None, None

# Create global instance
db_manager = DatabaseManager()
