import logging
import json
import re

logger = logging.getLogger(__name__)

# Fields to extract for compensation plans - customize as needed
PLAN_FIELDS = [
    "plan_name",
    "plan_period",
    "territory",
    "quota",
    "commission_structure",
    "tiers",
    "bonus_rules",
    "sales_target",
    "effective_dates"
]

def extract_plan_struct_from_text(plan_paragraph: str, llm_complete_func) -> dict:
    """
    Extract structured plan info from a business description using LLM.
    Returns: dict with extracted fields, missing fields, and original text.
    """
    fields_str = ", ".join(PLAN_FIELDS)
    prompt = (
        f"Extract the following fields from this compensation plan description if present: {fields_str}. "
        f"Respond ONLY as valid JSON in the format: {{'extracted': {{field:value,...}}, 'missing_fields': [field,...]}}. "
        f"Do NOT include markdown formatting, code blocks, or any text before or after the JSON. "
        f"Start your response with {{ and end with }}. "
        f"User plan description:\n{plan_paragraph}\n"
    )
    try:
        resp = llm_complete_func(prompt)
        
        # ✅ FIX: Clean the response before parsing
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', resp)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        # Extract just the JSON part (from first { to last })
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        
        if start != -1 and end > start:
            json_str = cleaned[start:end]
        else:
            json_str = cleaned
        
        # Log what we're about to parse (for debugging)
        logger.info(f"[NLP] Attempting to parse JSON: {json_str[:200]}...")
        
        data = json.loads(json_str)
        
        # Defensive parsing in case model returns plain {} or only fields
        extracted = data.get('extracted', {}) if isinstance(data, dict) else {}
        missing_fields = data.get('missing_fields', []) if isinstance(data, dict) else PLAN_FIELDS
        
        logger.info(f"[NLP] Successfully extracted: {list(extracted.keys())}")
        
        return {
            "status": "ok",
            "extracted": extracted,
            "missing_fields": missing_fields,
            "original_text": plan_paragraph
        }
    except Exception as e:
        logger.error(f"NLP plan builder LLM extraction failed: {e}")
        logger.error(f"[NLP] Raw response was: {resp[:500]}")  # Log first 500 chars
        return {
            "status": "error",
            "error": str(e),
            "original_text": plan_paragraph,
            "extracted": {},
            "missing_fields": PLAN_FIELDS
        }
