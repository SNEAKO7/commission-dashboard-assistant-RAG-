
#############################################################################################################################################


import json
import logging
import re

logger = logging.getLogger(__name__)

def is_plan_creation_intent(user_msg, llm_complete_func):
    """
    Determine if user wants to create a new plan vs query existing data
    """
    msg = user_msg.lower().strip()
    
    # STEP 1: Check for PLAN CREATION keywords FIRST (highest priority)
    plan_keywords = [
        "create plan", "create a plan", "create new plan", "start plan", "new plan", 
        "commission plan", "setup plan", "plan creation", "add plan", "build plan", 
        "comp plan", "make plan", "design plan", "configure plan"
    ]
    
    logger.info(f"[DEBUG-INTENT] [KEYWORD PHASE] Checking keywords in message: {msg}")
    
    for keyword in plan_keywords:
        if keyword in msg:
            logger.info(f"[DEBUG-INTENT] Plan creation keyword matched: '{keyword}'")
            return True
    
    # STEP 2: Check for DATABASE QUERY patterns with WORD BOUNDARIES
    # This prevents false positives like "all" matching in "called"
    db_query_patterns = [
        r'\bshow\b', r'\blist\b', r'\bdisplay\b', r'\bget\b', r'\bfind\b', r'\bsearch\b',
        r'\bwhat are\b', r'\bwhat is\b', r'\bhow many\b', r'\bcount\b', r'\ball\b',
        r'\bactive plans\b', r'\bactive programs\b', r'\bcurrent plans\b', 
        r'\bexisting plans\b', r'\bview plans\b', r'\bsee plans\b'
    ]
    
    logger.info(f"[DEBUG-INTENT] [DATABASE CHECK] Checking for DB patterns in: {msg}")
    
    for pattern in db_query_patterns:
        match = re.search(pattern, msg)
        if match:
            matched_text = match.group()
            logger.info(f"[DEBUG-INTENT] Database pattern found: '{matched_text}' - NOT plan creation")
            return False
    
    # STEP 3: LLM fallback with better context
    meta_prompt = f"""You are helping distinguish between two types of user requests:

                    TYPE A: VIEWING/QUERYING existing data (show, list, find, get, what are...)
                    TYPE B: CREATING/BUILDING new compensation plans (create, make, build, design...)

                    Is this message asking to CREATE or BUILD a NEW compensation/commission plan?

                    IMPORTANT: 
                    - If the message asks to VIEW, SHOW, LIST, FIND, or GET existing data, answer 'False'
                    - Only answer 'True' if the user wants to CREATE/BUILD something new

                    Reply ONLY 'True' or 'False'.

                    Message: {user_msg}"""
    
    logger.info(f"[DEBUG-INTENT] [LLM PHASE] Sending improved meta prompt to Claude")
    
    try:
        resp = llm_complete_func(prompt=meta_prompt, max_tokens=10, temperature=0)
        logger.info(f"[DEBUG-INTENT] LLM returned: {resp}")
        result = resp.strip().lower().startswith("true")
        logger.info(f"[DEBUG-INTENT] Final intent decision: {result}")
        return result
    except Exception as e:
        logger.error(f"[DEBUG-INTENT] LLM intent detection failed: {e}")
        return False

# plan extraction helper
def extract_plan_fields_claude(user_msg, llm_complete_func):
    required_fields = ["plan_name", "period", "territory", "quota", "commission_rate", "sales_target"]
    
    prompt = f'''
                Extract the following fields from this paragraph if present: plan_name, period, territory, quota, commission_rate, sales_target. Answer as JSON:

                User description: """{user_msg}"""

                Respond as:
                {{
                "extracted": {{}},
                "missing_fields": [ ... ]
                }}
                '''
    
    resp = llm_complete_func(prompt=prompt, max_tokens=400, temperature=0)
    
    try:
        out = json.loads(resp)
        return out.get("extracted", {}), out.get("missing_fields", [])
    except Exception:
        return {}, required_fields
