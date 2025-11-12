import logging
import json

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
        f"User plan description:\n{plan_paragraph}\n"
    )
    try:
        resp = llm_complete_func(prompt) #, maxtokens=600, temperature=0)
        data = json.loads(resp)
        # Defensive parsing in case model returns plain {} or only fields
        extracted = data.get('extracted', {}) if isinstance(data, dict) else {}
        missing_fields = data.get('missing_fields', []) if isinstance(data, dict) else PLAN_FIELDS
        return {
            "status": "ok",
            "extracted": extracted,
            "missing_fields": missing_fields,
            "original_text": plan_paragraph
        }
    except Exception as e:
        logger.error(f"NLP plan builder LLM extraction failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "original_text": plan_paragraph,
            "extracted": {},
            "missing_fields": PLAN_FIELDS
        }
