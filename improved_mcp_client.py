"""
Improved MCP Client - Wrapper for backward compatibility
"""
import logging
from mcp_client import mcp_client

logger = logging.getLogger(__name__)

def execute_query(user_query: str, context: str = "") -> dict:
    """Execute natural language query using MCP + Claude"""
    return mcp_client.process_natural_language_query(user_query, context)

def format_query_response(result: dict) -> str:
    """Format the response for display"""
    if not result.get('success'):
        return f"Sorry, I couldn't retrieve the information: {result.get('error', 'Unknown error')}"
    
    # Return the pre-formatted English response from Claude
    formatted = result.get('formatted_response')
    if formatted:
        return formatted
    
    # Fallback formatting
    data = result.get('data', [])
    if not data:
        return "I couldn't find any data matching your query."
    
    if len(data) == 1:
        row = data[0]
        parts = [f"{k.replace('_', ' ')}: {v}" for k, v in row.items() if v is not None]
        return "Here's what I found: " + ", ".join(parts)
    
    lines = [f"I found {len(data)} results:"]
    for i, row in enumerate(data[:5], 1):
        summary = ", ".join([f"{k.replace('_', ' ')}: {v}" for k, v in row.items() 
                           if v is not None and k in ['program_name', 'plan_type', 'status']])
        lines.append(f"{i}. {summary}")
    
    if len(data) > 5:
        lines.append(f"... and {len(data) - 5} more results")
    
    return "\n".join(lines)

# Make mcp_client accessible
mcp_client = mcp_client
