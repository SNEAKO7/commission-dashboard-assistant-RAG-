#!/usr/bin/env python3
"""
SQL Security Validator - Ensures only SELECT queries are executed
"""

import re
from typing import Tuple, Optional

def validate_sql_security(sql: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate SQL query for security (only SELECT allowed)
    
    Returns:
        (is_valid, reason, cleaned_query)
    """
    if not sql or not sql.strip():
        return False, "Empty query", None
    
    sql = sql.strip()
    sql_upper = sql.upper()
    
    # Remove trailing semicolon
    if sql.endswith(';'):
        sql = sql[:-1]
    
    # Check if it starts with SELECT
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT queries allowed", None
    
    # Check for forbidden keywords
    forbidden = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE'
    ]
    
    for keyword in forbidden:
        if re.search(r'\b' + keyword + r'\b', sql_upper):
            return False, f"Forbidden keyword: {keyword}", None
    
    # Check for multiple statements
    if ';' in sql:
        return False, "Multiple statements not allowed", None
    
    return True, "Query is safe", sql
