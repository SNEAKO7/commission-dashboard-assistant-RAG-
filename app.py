import os
import json
import logging
import threading
import urllib3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, send_from_directory
from flask_session import Session
from dotenv import load_dotenv
import requests
#from final_mcp_client import mcp_client
from improved_mcp_client import execute_query, format_query_response
from improved_mcp_client import mcp_client
import jwt
from functools import wraps
from flask_cors import CORS
import psycopg2
import time
from mcp_client import mcp_client
import intent_detection
from nlp_plan_builder import extract_plan_struct_from_text
from datetime import datetime, timedelta
import re



# Import RAG functionality
from rag import retrieve_context, clear_vector_cache

#for redis 
from redis_chat_session import save_message, get_chat_history, clear_chat_history
import redis

#for prediction analysis
from ml_handler import handle_ml_request
from intent_detection import is_ml_prediction_intent, is_ml_optimization_intent, extract_ml_params
from plan_editor import plan_editor

# --- Setup & config ---
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, origins=["http://localhost:9000"], supports_credentials=True)
secret = os.getenv("FLASK_SECRET_KEY", None)
if secret:
    app.secret_key = secret
else:
    # fallback random each run (restart clears sessions)
    app.secret_key = os.urandom(24)

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True #False
app.config["SESSION_USE_SIGNER"] = True

# ========== ML SYSTEM AUTO-INITIALIZATION ==========
logger.info("🤖 Initializing ML Prediction System...")
try:
    from ml_advanced_predictor import tenant_predictor
    logger.info("✅ ML Predictor loaded and ready")
except Exception as e:
    logger.error(f"❌ ML Predictor initialization failed: {e}")
# ====================================================

app.permanent_session_lifetime = timedelta(hours=2)
Session(app)

API_BASE_URL = os.getenv("BACKEND_API_BASE_URL", "https://localhost:8081")
JWT_TOKEN = os.getenv("JWT_TOKEN", "")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "documents")  # Folder path for RAG documents

logger.info(f"✅ JWT_TOKEN present: {bool(JWT_TOKEN)}")
logger.info(f"🔧 API_BASE_URL: {API_BASE_URL}")
logger.info(f"📂 DOCUMENTS_FOLDER: {DOCUMENTS_FOLDER}")

def get_client_id_for_org(org_id):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "commissions"),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASSWORD", "password"),
        port=int(os.getenv("DB_PORT", 5432)),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT parent_id FROM organization WHERE status=1 AND id=%s ORDER BY id DESC LIMIT 1", (org_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

def verify_jwt_token(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        try:
            # Replace 'your-jwt-secret' with the secret used by your .NET backend!
            #payload = jwt.decode(token, 'your-jwt-secret', algorithms=['HS256'])
            payload = jwt.decode(token, "V_E_R_Y_S_E_C_R_E_T_K_E_Y_Commission_WEB_APP", algorithms=['HS256'], audience="http://callippus.co.uk", issuer="http://callippus.co.uk")
            print("Decoded JWT payload:", payload)
            session['jwt_token'] = token
            
            # Save user fields needed for multitenancy:
            request.org_id    = payload.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/spn")
            request.org_code  = payload.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/stateorprovince")
            request.user_id   = payload.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier")
            request.username  = payload.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name")

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated


# --- Server-side persistent user-state (optional keyed by user_id) ---
USER_STATE = {}
USER_STATE_LOCK = threading.Lock()

def load_user_state(user_id):
    with USER_STATE_LOCK:
        s = USER_STATE.get(user_id)
        return dict(s) if isinstance(s, dict) else {}

def save_user_state(user_id, state_dict):
    with USER_STATE_LOCK:
        USER_STATE[user_id] = dict(state_dict)

def clear_user_state(user_id):
    with USER_STATE_LOCK:
        if user_id in USER_STATE:
            del USER_STATE[user_id]

# --- INTENT RECOGNITION ---

def is_plan_creation_intent(user_msg, llm_complete_func):
    msg = user_msg.lower().strip()
    plan_keywords = [
        "create plan", "create a plan", "start plan", "new plan", "commission plan",
        "setup plan", "plan creation", "add plan", "build plan"
    ]
    for keyword in plan_keywords:
        if keyword in msg:
            logger.info(f"🎯 Plan creation intent detected: '{keyword}' found in message")
            return True
    meta_prompt = f"""Is this message from a user a description of a new compensation plan or an attempt to create a compensation plan? Answer True or False: {user_msg}"""
    try:
        resp = llm_complete_func(prompt=meta_prompt, max_tokens=2, temperature=0)
        logger.info(f'[intent LLM] LLM intent response: {resp}')
        return resp.strip().lower().startswith("true")
    except Exception as e:
        return False


def is_database_query_intent(message: str) -> bool:
    """
    Determine if the user wants to query the database
    """
    if not is_database_enabled():
        return False
    
    message_lower = message.lower().strip()
    
    # Database query keywords
    db_keywords = [
        "show", "list", "find", "get", "what", "how many", "count",
        "total", "sum", "average", "when", "who", "active plan", "current plan",
        "commission", "payment", "employee", "program", "rule", "condition",
        "tier", "range", "schedule", "assignment", "paid", "amount"
    ]
    
    # Check if message contains database entities (use schema)
    try:
        schema = mcp_client.get_database_schema()
        if schema:
            # Check for table names in message
            for table_name in schema.keys():
                table_words = table_name.replace('_', ' ').split()
                for word in table_words:
                    if word in message_lower and len(word) > 3:
                        logger.info(f"🎯 Database query intent detected: table '{table_name}' reference found")
                        return True
    except Exception as e:
        logger.debug(f"Schema check failed in intent detection: {e}")
    
    # Check for database query keywords
    for keyword in db_keywords:
        if keyword in message_lower:
            # Make sure it's not a plan creation intent
            # if not is_plan_creation_intent(message):
            if not intent_detection.is_plan_creation_intent(message, mcp_client.claude_complete):
                logger.info(f"🎯 Database query intent detected: '{keyword}' found in message")
                return True
    
    return False

def is_database_enabled():
    try:
        from improved_mcp_client import execute_query
        return True
    except ImportError:
        return False

def handle_database_query(user_message: str, offset=0) -> str:
    """
    Handle database queries using improved MCP client 
    """
    try:
        # Execute the natural language query using the improved MCP/NL→SQL flow
        org_id = getattr(request, 'org_id', None)
        org_code = getattr(request, 'org_code', None)
        client_id = get_client_id_for_org(org_id) if org_id else None
        result = execute_query(user_message, org_id=org_id, client_id=client_id, offset=offset)

        # Format for pretty printing, safe for end user
        formatted_response = format_query_response(result)
        return formatted_response

    except Exception as e:
        logger.error(f"❌ [DB] Error processing database query: {e}")
        error_text = str(e)
        if ("429" in error_text or "rate_limit_error" in error_text or "Too Many Requests" in error_text):
            return "The system is currently handling too many requests to the AI service. Please wait a few seconds and try again."
        return "The server is restarting or temporarily unavailable. Please try again in a few seconds."
        
# --- Pagination helpers ---

def get_plan_offset():
    return session.get('plan_offset', 0)

def set_plan_offset(offset):
    session['plan_offset'] = offset
    session.modified = True

def increment_plan_offset(n=10):
    session['plan_offset'] = get_plan_offset() + n
    session.modified = True

def reset_plan_offset():
    session['plan_offset'] = 0
    session.modified = True



# --- RAG QUERY HANDLER - SIMPLIFIED SINCE rag.py NOW HANDLES CLEAN RESPONSES ---
def handle_rag_query(user_message: str) -> str:
    """
    Handle general queries using RAG functionality
    """
    logger.info(f"🧠 [RAG] Processing query: '{user_message}'")
    
    try:
        # retrieve_context now returns clean answer directly
        clean_answer, chunks = retrieve_context(user_message, DOCUMENTS_FOLDER, k=7)
        
        if not clean_answer:
            logger.warning("📭 [RAG] No relevant context found")
            return "I couldn't find any relevant information in the documents. Please make sure the documents are properly loaded or try rephrasing your question."
        
        logger.info(f"📋 [RAG] Retrieved answer from {len(chunks)} chunks")
        logger.info(f"✅ [RAG] Response generated: {len(clean_answer)} characters")
        
        return clean_answer
        
    except Exception as e:
        logger.error(f"❌ [RAG] Error processing query: {e}")
        error_text = str(e)
        if "429" in error_text or "rate_limit_error" in error_text or "Too Many Requests" in error_text:
            return "The system is currently handling too many requests to the AI service. Please wait a few seconds and try again."
        return f"Sorry, I encountered an error while searching the documents: {str(e)}"


def api_headers():
    print("[DEBUG] jwt_token IN api_headers:", session.get('jwt_token'))
    h = {"Accept": "application/json"}
    token = session.get("jwt_token")
    if not token:
        token = JWT_TOKEN  # fallback ONLY if not in session (dev/testing)
    if token:
        h["Authorization"] = f"Bearer {token}"
    print("API HEADERS USED FOR FETCH:", h)
    return h


def fetch_json_safe(url, name_for_logs=None, timeout=8):
    name = name_for_logs or url
    try:
        logger.info(f"[{name}] GET {url}")
        r = requests.get(url, headers=api_headers(), timeout=timeout, verify=False)
        logger.info(f"[{name}] Status: {r.status_code}")
        txt = (r.text or "")[:1000]
        logger.debug(f"[{name}] Body (truncated): {txt}")
        
        if r.status_code != 200:
            return None
        
        ct = (r.headers.get("Content-Type") or "").lower()
        if "json" in ct or r.text.strip().startswith(("[", "{")):
            try:
                return r.json()
            except Exception as e:
                logger.error(f"[{name}] JSON parse error: {e}")
                logger.debug(r.text[:2000])
                return None
        else:
            logger.warning(f"[{name}] Non-json response")
            return None
    except Exception as e:
        logger.error(f"[{name}] Request failed: {e}")
        return None

# --- API fetchers (use your endpoints) ---
def fetch_schedules():
    return fetch_json_safe(f"{API_BASE_URL}/api/PlanCreationWebApi/GetCommissionSchedule", "fetch_schedules") or []

def fetch_assignee_objects():
    return fetch_json_safe(f"{API_BASE_URL}/api/PlanCreationWebApi/GetAssigneeNameList", "fetch_assignee_objects") or []

def fetch_plan_types():
    return fetch_json_safe(f"{API_BASE_URL}/api/PlanTypeMasterWebApi/GetPlanTypeMasterList", "fetch_plan_types") or []

def fetch_plan_params():
    # attempt endpoint; fallback to five if not available
    res = fetch_json_safe(f"{API_BASE_URL}/api/PlanCreationWebApi/GetPlanParams", "fetch_plan_params")
    if isinstance(res, list) and res:
        # try extract names
        out = []
        for r in res:
            if isinstance(r, str):
                out.append(r)
            elif isinstance(r, dict):
                name = r.get("name") or r.get("paramName") or r.get("param")
                if name:
                    out.append(name)
        if out:
            return out
    return ["Territory", "Product", "BusinessPartner", "Plant", "Group"]

def fetch_uom():
    return fetch_json_safe(f"{API_BASE_URL}/api/PlanCreationWebApi/GetUomList", "fetch_uom") or []

def fetch_currency():
    return fetch_json_safe(f"{API_BASE_URL}/api/PlanCreationWebApi/GetCurrencyList", "fetch_currency") or []

# --- Hardcoded lists ---
OBJECT_TYPES = ["Invoices", "Contracts", "Sales Orders"]
CATEGORY_TYPES = ["Flat", "Slab", "Tiered"]
RANGE_TYPES = ["Amount", "Quantity"]
BASE_VALUES = ["Gross", "Net"]
PLAN_BASES = ["Revenue", "Margin"]
VALUE_TYPES = ["Percentage", "Amount"]
YES_NO = ["Yes", "No"]
ON_OFF = ["On", "Off"]

# --- Helpers for presenting & selecting numbered lists ---
def present_numbered_list(items, heading=None):
    """
    Returns a single string with numbered lines and the options list saved for later resolution.
    """
    if heading is None:
        heading = ""
    lines = []
    if heading:
        lines.append(heading)
    if not items:
        lines.append("(no options available)")
        return "\n".join(lines)
    
    for i, it in enumerate(items, start=1):
        # if item is dict try nice label
        if isinstance(it, dict):
            label = it.get("schedulerName") or it.get("planName") or it.get("objectType") or it.get("label") or it.get("currencyCode") or it.get("currencyName") or str(it)
        else:
            label = str(it)
        lines.append(f"{i}. {label}")
    
    return "\n".join(lines)

def resolve_choice(choice, options):
    """
    Accepts choice: number (string) or text. Returns resolved label (string) or None.
    options: list of strings or dicts (we return pretty label as string)
    """
    if not choice:
        return None
    
    c = choice.strip()
    
    # numeric
    if c.isdigit():
        idx = int(c) - 1
        if 0 <= idx < len(options):
            opt = options[idx]
            return option_label(opt)
        return None
    
    # exact match case-insensitive against labels
    for opt in options:
        lbl = option_label(opt)
        if lbl.lower() == c.lower():
            return lbl
    
    # startswith fallback
    for opt in options:
        lbl = option_label(opt)
        if lbl.lower().startswith(c.lower()):
            return lbl
    
    return None

def option_label(opt):
    if isinstance(opt, dict):
        return opt.get("schedulerName") or opt.get("planName") or opt.get("objectType") or opt.get("label") or opt.get("currencyCode") or opt.get("currencyName") or str(opt)
    return str(opt)

# --- Combination helper: fetch values for a param from assignee objects as colleague described ---
def get_param_options_for_combination(param_name, assignee_objs):
    if not param_name:
        return []
    
    out = []
    valid_tables = set()
    
    for a in assignee_objs:
        tname = (a.get("tableName") or "").strip().lower()
        if tname:
            valid_tables.add(tname)
    
    for a in assignee_objs:
        try:
            if str(a.get("type") or "").strip().lower() == param_name.strip().lower() and str(a.get("subType") or "").strip().lower() != "sales team":
                tname = (a.get("tableName") or "").strip().lower()
                if tname in valid_tables:
                    out.append({"label": a.get("objectType") or a.get("label") or a.get("objectName") or a.get("object") or a.get("name") or "", "id": a.get("id"), "tableName": a.get("tableName")})
        except Exception as e:
            logger.debug(f"[get_param_options_for_combination] error: {e}")
    
    # dedupe by label preserving order
    seen = set(); deduped = []
    for o in out:
        lbl = (o.get("label") or "").strip()
        if not lbl: continue
        if lbl.lower() in seen: continue
        seen.add(lbl.lower()); deduped.append(o)
    
    return deduped

# --- Build and store options once when starting plan creation ---
def build_options_and_store():
    opts = {}
    opts["schedules"] = fetch_schedules() # schedule objects with schedulerName
    opts["assignee_objects"] = fetch_assignee_objects() # raw assignee objects
    
    # extract readable assignee names for selection (filter Sales Team as colleague mentioned for assignee dropdown? keep all readable)
    assignees = []
    for a in opts["assignee_objects"]:
        label = a.get("objectType") or a.get("label") or a.get("objectName") or a.get("name")
        if label:
            assignees.append(label)
    opts["assignees"] = sorted(list(dict.fromkeys(assignees))) # dedupe but preserve order
    
    opts["plan_types"] = [p.get("planName") for p in fetch_plan_types()] or []
    opts["plan_params"] = fetch_plan_params()
    opts["uom"] = fetch_uom()
    opts["currency"] = fetch_currency()
    opts["object_types"] = OBJECT_TYPES
    opts["category_types"] = CATEGORY_TYPES
    opts["range_types"] = RANGE_TYPES
    opts["base_values"] = BASE_VALUES
    opts["plan_bases"] = PLAN_BASES
    opts["value_types"] = VALUE_TYPES
    
    return opts

# --- POST to create program/plans (final submit) - UPDATED WITH BETTER ERROR HANDLING ---

def post_program_creation(plan_payload):
    url = f"{API_BASE_URL}/api/PlanCreationWebApi/PostProgramCreation"
    try:
        # Check if this is a simple payload (from NLP) or complex (from wizard)
        if 'plan_name' in plan_payload:
            # Convert simple NLP payload to API format
            plan_payload = build_simple_payload_from_plan_data(plan_payload)
        
        logger.info("[post_program_creation] posting to URL: " + url)
        logger.info("[post_program_creation] payload keys: " + str(list(plan_payload.keys())))
        logger.info("[post_program_creation] posting payload (truncated): " + json.dumps(plan_payload)[:1500])
        
        headers = api_headers()
        headers["Content-Type"] = "application/json"
        
        r = requests.post(url, json=plan_payload, headers=headers, timeout=30, verify=False)
        logger.info(f"[post_program_creation] status: {r.status_code}")
        logger.info(f"[post_program_creation] response headers: {dict(r.headers)}")
        logger.info(f"[post_program_creation] response text: {r.text[:1000]}")
        
        if r.status_code == 404:
            logger.error("[post_program_creation] 404 error - endpoint not found")
            return {"error": "API endpoint not found (404). Check server configuration.", "status_code": 404}
        elif r.status_code == 401:
            logger.error("[post_program_creation] 401 error - authentication failed")
            return {"error": "Authentication failed (401). Check JWT token.", "status_code": 401}
        elif r.status_code == 500:
            logger.error("[post_program_creation] 500 error - server error")
            return {"error": f"Server error (500): {r.text[:200]}", "status_code": 500}
        elif r.status_code not in [200, 201]:
            logger.warning(f"[post_program_creation] unexpected status: {r.status_code}")
            return {"error": f"Unexpected response status: {r.status_code}", "status_code": r.status_code, "response": r.text}
        
        try:
            return r.json()
        except:
            # Success but non-JSON response
            if r.status_code in [200, 201]:
                return {"success": True, "status_code": r.status_code, "text": r.text[:2000]}
            return {"status_code": r.status_code, "text": r.text[:2000]}
    except Exception as e:
        logger.error(f"[post_program_creation] error: {e}")
        return {"error": str(e)}

# --- Start plan creation: initialize session keys ---
def start_plan_creation_session():
    opts = build_options_and_store()
    
    if not opts["schedules"] or not opts["assignees"] or not opts["plan_types"]:
        # allow flow but warn if critical masters are missing
        logger.warning("[start_plan_creation_session] some master data missing; check backend endpoints")
    
    # session.clear()
    jwt = session.get("jwt_token")  # save JWT
    session.clear()
    if jwt:
        session["jwt_token"] = jwt  # restore JWT
    session["mode"] = "plan_creation"
    session["phase"] = 1
    session["stage"] = "plan_name"
    session["options"] = opts
    session["plan_data"] = {
        "plan_name": None,
        "calculation_schedule": None,
        "payment_schedule": None,
        "assignee_name": None,
        "object_type": None,
        "valid_from": None,
        "valid_to": None,
        "plan_rules": []
    }
    session["current_rule"] = None
    session["current_assignment"] = None
    session["awaiting_final_confirmation"] = False
    session["org_id"]   = getattr(request, "org_id", None)
    session["org_code"] = getattr(request, "org_code", None)
    session["user_id"]  = getattr(request, "user_id", None)
    session["username"] = getattr(request, "username", None)
    # (client_id/client_code only if needed and correctly mapped)

    session.modified = True

# --- Flow handler (core) - COMPLETE PLAN CREATION LOGIC ---

def handle_message_in_flow(user_msg):
    """
    Handle wizard continuation - process user input in plan creation mode
    Returns: dict with 'response' key
    """
    logger.info(f"[WIZARD] Processing continuation message: {user_msg}")
    
    try:
        # Get current plan data
        plan_data = session.get('plan_data', {})
        logger.info(f"[WIZARD] Current plan data: {plan_data}")
        
        # Check if this is NLP-based plan creation (has plan_data)
        if plan_data:
            # NLP-based continuation
            user_msg_lower = user_msg.lower().strip()
            
            # Check if user wants to submit
            if any(word in user_msg_lower for word in ['submit', 'save']):
                # Check if we have minimum required fields
                required_fields = ['plan_name']
                missing_required = [f for f in required_fields if not plan_data.get(f)]
                
                if missing_required:
                    return {
                        "response": f"Cannot submit yet. Plan name is required. Please provide a plan name first."
                    }
                
                # SHOW COMPLETE SUMMARY with ALL fields (including empty)
                logger.info("[WIZARD] User requested submit - showing final confirmation with ALL fields")
                session['awaiting_final_confirmation'] = True
                session.modified = True
                
                #if user_id:
                    #save_user_state(user_id, dict(session))
                
                # Generate complete summary showing ALL fields
                complete_summary = format_plan_summary(plan_data, show_all_fields=True)
                
                return {
                    "response": complete_summary
                }

            # Check for FINAL CONFIRMATION (SECOND STEP: Actually save)
            if user_msg_lower.strip() == 'confirm' and session.get('awaiting_final_confirmation'):
                logger.info("[WIZARD] User confirmed - proceeding with actual save")
                
                # Clear the confirmation flag
                session['awaiting_final_confirmation'] = False
                session.modified = True
                
                # Build and submit the plan
                try:
                    # Log what we're about to save
                    logger.info(f"[WIZARD] Saving plan to database with data: {json.dumps(plan_data, indent=2)}")
                    
                    # Use your EXISTING API submission
                    resp = post_program_creation(plan_data)
                    
                    # Clear session on success
                    if isinstance(resp, dict) and not resp.get("error"):
                        jwt = session.get("jwt_token")
                        session.clear()
                        if jwt:
                            session["jwt_token"] = jwt
                        
                        return {
                            "response": f"✅ **SUCCESS!** \n\nYour plan '{plan_data.get('plan_name')}' has been created successfully!\n\nYou can now create another plan or ask me questions about existing plans."
                        }
                    else:
                        error_msg = resp.get('error', 'Unknown error') if isinstance(resp, dict) else str(resp)
                        return {
                            "response": f"❌ Error saving plan: {error_msg}\n\nPlease check the details and try again, or type 'edit' to modify the plan."
                        }
                        
                except Exception as e:
                    logger.error(f"[WIZARD] Error submitting plan: {e}")
                    return {
                        "response": f"❌ Error saving plan: {str(e)}\n\nPlease try again or contact support."
                    }

            # Handle cancel during confirmation
            if user_msg_lower.strip() == 'cancel' and session.get('awaiting_final_confirmation'):
                session['awaiting_final_confirmation'] = False
                session.modified = True
                
                #if user_id:
                    #save_user_state(user_id, dict(session))
                
                return {
                    "response": "Plan submission cancelled. You can continue editing or type 'submit' again when ready."
                }
            ''' if any(word in user_msg_lower for word in ['submit', 'save', 'confirm', 'yes', 'looks good', 'correct']):
                # Check if we have minimum required fields
                required_fields = ['plan_name']  # Minimum required
                missing_required = [f for f in required_fields if not plan_data.get(f)]
                
                if missing_required:
                    return {
                        "response": f"Cannot submit yet. Plan name is required. Please provide a plan name first."
                    }
                
                # Build and submit the plan
               # Build and submit the plan
                try:
                    # Use your EXISTING API submission (NO hardcoded values!)
                    resp = post_program_creation(plan_data)
                    
                    # Clear session on success
                    if isinstance(resp, dict) and not resp.get("error"):
                        jwt = session.get("jwt_token")
                        session.clear()
                        if jwt:
                            session["jwt_token"] = jwt
                        
                        return {
                            "response": f"✅ **SUCCESS!** \n\nYour plan '{plan_data.get('plan_name')}' has been created successfully!\n\nYou can now create another plan or ask me questions about existing plans."
                        }
                    else:
                        error_msg = resp.get('error', 'Unknown error') if isinstance(resp, dict) else str(resp)
                        return {
                            "response": f"❌ Error saving plan: {error_msg}\n\nPlease check the details and try again, or type 'edit' to modify the plan."
                        }
                        
                except Exception as e:
                    logger.error(f"[WIZARD] Error submitting plan: {e}")
                    return {
                        "response": f"❌ Error saving plan: {str(e)}\n\nPlease try again or contact support."
                    }'''
 

            ''' # Build and submit the plan
                try:
                    # REAL DATABASE SUBMISSION (replace the fake one)
                    logger.info(f"[WIZARD] Submitting plan to database: {plan_data}")
                    
                    # Convert plan data to database format
                    db_plan = {
                        "program_name": plan_data.get('plan_name'),
                        "org_id": 94,  # Your org ID
                        "client_id": 93,  # Your client ID  
                        "status": 1,  # Active
                        "valid_from": "2025-01-01",  # Extract from effective_dates
                        "valid_to": "2025-12-31",    # Extract from effective_dates
                        "assignee_name": "System",   # Default assignee
                        "table_name": "commission_plans",
                        "object_type": "Plan",
                        "created_by": user_id or "system",
                        "created_date": datetime.now().isoformat(),
                        "plan_details": json.dumps(plan_data)  # Store full plan as JSON
                    }
                    
                    # Insert into plan_master table using your database connection
                    import psycopg2
                    import json
                    from datetime import datetime
                    
                    # Use your existing database connection
                    conn = psycopg2.connect(
                        host=os.getenv('DB_HOST'),
                        database=os.getenv('DB_NAME'), 
                        user=os.getenv('DB_USER'),
                        password=os.getenv('DB_PASSWORD')
                    )
                    
                    cursor = conn.cursor()
                    
                    insert_query = """
                        INSERT INTO plan_master (
                            program_name, org_id, client_id, status, valid_from, valid_to,
                            assignee_name, table_name, object_type, created_by, created_date, plan_details
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """
                    
                    cursor.execute(insert_query, (
                        db_plan["program_name"],
                        db_plan["org_id"], 
                        db_plan["client_id"],
                        db_plan["status"],
                        db_plan["valid_from"],
                        db_plan["valid_to"],
                        db_plan["assignee_name"],
                        db_plan["table_name"],
                        db_plan["object_type"],
                        db_plan["created_by"],
                        db_plan["created_date"],
                        db_plan["plan_details"]
                    ))
                    
                    plan_id = cursor.fetchone()[0]
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    logger.info(f"[WIZARD] Plan saved successfully with ID: {plan_id}")
                    
                    # Clear session on success
                    jwt = session.get("jwt_token")
                    session.clear()
                    if jwt:
                        session["jwt_token"] = jwt
                    
                    return {
                        "response": f"✅ **SUCCESS!** \n\nYour plan '{plan_data.get('plan_name')}' has been created successfully with ID {plan_id}!\n\nYou can now create another plan or ask me questions about existing plans."
                    }
                        
                except Exception as e:
                    logger.error(f"[WIZARD] Error submitting plan to database: {e}")
                    return {
                        "response": f"❌ Error saving plan to database: {str(e)}\n\nPlease try again or contact support."
                    }'''


            # Check if user wants to edit
            if any(word in user_msg_lower for word in ['edit', 'change', 'modify', 'update']):
                # If just "edit" with no details, ask what to change
                if user_msg_lower.strip() in ['edit', 'modify', 'change', 'update']:
                    return {
                        "response": "What would you like to change? Please specify the field and new value.\n\nExample: 'Change plan name to Q4 Sales Champions' or 'Update commission to 10%'"
                    }
                
                # Try to extract field and value from edit commands
                field_updated = False
                
                # Improved patterns to handle multi-word field names
                patterns = [
                    r'change\s+(?:the\s+)?(.+?)\s+to\s+(.+)',  # "change [the] plan name to X"
                    r'update\s+(?:the\s+)?(.+?)\s+to\s+(.+)',  # "update [the] commission to X"
                    r'(.+?)\s*=\s*(.+)',                        # "plan_name = X"
                    r'(.+?)\s*:\s*(.+)'                         # "plan_name: X"
                ]
                
                import re
                for pattern in patterns:
                    match = re.search(pattern, user_msg_lower)
                    if match:
                        raw_field = match.group(1).strip()
                        value = match.group(2).strip()
                        
                        # Clean up field name - remove articles and extra spaces
                        raw_field = raw_field.replace('the ', '').replace(' ', '_')
                        
                        # Map common field variations to actual field names
                        field_map = {
                            'name': 'plan_name',
                            'plan_name': 'plan_name',
                            'period': 'plan_period',
                            'plan_period': 'plan_period',
                            'commission': 'commission_structure',
                            'commission_structure': 'commission_structure',
                            'bonus': 'bonus_rules',
                            'bonus_rules': 'bonus_rules',
                            'target': 'sales_target',
                            'sales_target': 'sales_target',
                            'quota': 'quota',
                            'territory': 'territory',
                            'tiers': 'tiers',
                            'effective_dates': 'effective_dates'
                        }
                        
                        field = field_map.get(raw_field)
                        
                        if field:
                            # Store old value for display
                            old_value = plan_data.get(field, 'Not set')
                            
                            # Update the field based on type
                            if field in ['quota', 'sales_target']:
                                # Convert to integer for numeric fields
                                try:
                                    plan_data[field] = int(value.replace(',', '').replace('$', ''))
                                except:
                                    plan_data[field] = value
                            else:
                                # For text fields, preserve original case from user input
                                # Extract the actual value from original message (not lowercase)
                                original_match = re.search(pattern.replace('(.+?)', '(.+?)').replace('(.+)', '(.+)'), user_msg, re.IGNORECASE)
                                if original_match:
                                    plan_data[field] = original_match.group(2).strip()
                                else:
                                    plan_data[field] = value
                            
                            session['plan_data'] = plan_data
                            session.modified = True
                            field_updated = True
                            
                            logger.info(f"[WIZARD] Updated {field} from '{old_value}' to '{plan_data[field]}'")
                            
                            # Show updated summary
                            summary = format_plan_summary(plan_data)
                            return {
                                "response": f"✏️ Updated {field.replace('_', ' ')} from '{old_value}' to: {plan_data[field]}\n\n{summary}\n\n✅ Type 'submit' to save this plan, or 'edit' to make more changes."
                            }
                        else:
                            logger.warning(f"[WIZARD] Could not map field '{raw_field}' to a known field")
                            break
                
                if not field_updated:
                    # Couldn't parse the edit command
                    return {
                        "response": "I couldn't understand what you want to change. Please use format like:\n- 'Change plan name to Q4 Champions'\n- 'Change the commission to 10%'\n- 'Update quota to 500000'\n- 'plan_name = Q4 Champions'"
                    }

            # PRIMARY APPROACH: Try NLP extraction FIRST
            if not user_msg_lower.startswith(('change', 'update', 'edit')):
                logger.info(f"[WIZARD] Starting field extraction for: {user_msg}")
                
                # STEP 1: Try NLP extraction using Claude AI
                nlp_success = False
                try:
                    logger.info(f"[WIZARD] Attempting NLP extraction...")
                    result = extract_plan_struct_from_text(user_msg, mcp_client.claude_complete)
                    logger.info(f"[WIZARD] NLP extraction result: {result}")
                    
                    if result["status"] == "ok" and result["extracted"]:
                        # Merge extracted fields with existing plan data
                        for key, value in result["extracted"].items():
                            if key == 'structure':  # Skip bogus fields
                                continue
                            if value and str(value).strip():  # Only update non-empty values
                                plan_data[key] = value
                                logger.info(f"[WIZARD] NLP extracted {key}: {value}")
                        nlp_success = True
                        
                except Exception as e:
                    logger.error(f"[WIZARD] NLP extraction failed: {e}")
                
                # STEP 2: If NLP failed, fallback to regex patterns
                if not nlp_success:
                    logger.info(f"[WIZARD] NLP failed, using regex fallback...")
                    import re
                    
                    # Extract Q4 2025 type patterns
                    period_match = re.search(r'(Q[1-4]\s*\d{4})', user_msg, re.IGNORECASE)
                    if period_match:
                        plan_data['plan_period'] = period_match.group(1)
                        logger.info(f"[WIZARD-REGEX] Extracted period: {period_match.group(1)}")
                    
                    # Extract quota
                    quota_match = re.search(r'quota[\s:]*(\d+)', user_msg, re.IGNORECASE)
                    if quota_match:
                        plan_data['quota'] = int(quota_match.group(1))
                        logger.info(f"[WIZARD-REGEX] Extracted quota: {quota_match.group(1)}")
                    
                    # Extract target
                    target_match = re.search(r'target[\s:]*(\d+)', user_msg, re.IGNORECASE)
                    if target_match:
                        plan_data['sales_target'] = int(target_match.group(1))
                        logger.info(f"[WIZARD-REGEX] Extracted target: {target_match.group(1)}")
                    
                    # Extract plan name - improved patterns for "Ash_tst2 is plan name"
                    if 'plan name' in user_msg_lower or 'name is' in user_msg_lower:
                        name_patterns = [
                            r'([A-Za-z0-9_]+)\s+is\s+plan\s+name',  # "Ash_tst2 is plan name"
                            r'plan\s+name\s+is\s+([A-Za-z0-9_\s]+)',  # "plan name is XYZ"
                            r'name\s+is\s+([A-Za-z0-9_\s]+)',         # "name is XYZ"
                            r'called\s+([A-Za-z0-9_\s]+)',           # "called XYZ"
                        ]
                        for pattern in name_patterns:
                            name_match = re.search(pattern, user_msg, re.IGNORECASE)
                            if name_match:
                                plan_data['plan_name'] = name_match.group(1).strip()
                                logger.info(f"[WIZARD-REGEX] Extracted plan name: {plan_data['plan_name']}")
                                break
                    
                    # Extract territory - improved patterns for "territory is north"
                    if 'territory' in user_msg_lower:
                        territory_patterns = [
                            r'territory\s+is\s+([A-Za-z]+)',      # "territory is north"
                            r'and\s+territory\s+is\s+([A-Za-z]+)',  # "and territory is north"
                            r',\s*territory\s+([A-Za-z]+)',       # ", territory north"
                        ]
                        for pattern in territory_patterns:
                            territory_match = re.search(pattern, user_msg, re.IGNORECASE)
                            if territory_match:
                                plan_data['territory'] = territory_match.group(1).strip()
                                logger.info(f"[WIZARD-REGEX] Extracted territory: {plan_data['territory']}")
                                break
                    
                    # Extract tiered structure
                    if 'tiered' in user_msg_lower or '%' in user_msg:
                        tiers = []
                        tier_patterns = re.findall(r'(\d+)%\s*(below|above|for|at|over)?\s*(quota|target|\d+)?', user_msg, re.IGNORECASE)
                        for rate, condition, threshold in tier_patterns:
                            tier_info = {'rate': f"{rate}%"}
                            if condition:
                                if 'below' in condition.lower():
                                    tier_info['threshold'] = 'Below quota'
                                elif 'above' in condition.lower() or 'over' in condition.lower():
                                    tier_info['threshold'] = 'Above quota'
                            tiers.append(tier_info)
                        if tiers:
                            plan_data['tiers'] = tiers
                            logger.info(f"[WIZARD-REGEX] Extracted tiers: {tiers}")
                    
                    # Extract bonus rules
                    bonus_match = re.search(r'bonus\s+(\d+)\s+for\s+exceeding\s+(\d+)%?', user_msg, re.IGNORECASE)
                    if bonus_match:
                        plan_data['bonus_rules'] = f"${bonus_match.group(1)} bonus for exceeding {bonus_match.group(2)}% of target"
                        logger.info(f"[WIZARD-REGEX] Extracted bonus: {plan_data['bonus_rules']}")
                    
                    # Extract dates
                    date_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\s+(?:to|through|-)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\s+\d{4}'
                    date_match = re.search(date_pattern, user_msg, re.IGNORECASE)
                    if date_match:
                        date_str = date_match.group(0)
                        parts = re.split(r'\s+(?:to|through|-)\s+', date_str, re.IGNORECASE)
                        if len(parts) == 2:
                            plan_data['effective_dates'] = {
                                'start': '2025-01-01',
                                'end': '2025-03-31'
                            }
                            logger.info(f"[WIZARD-REGEX] Extracted dates: {plan_data['effective_dates']}")
                
                # Save updated plan data
                session["plan_data"] = plan_data
                session.modified = True
                
                # Show updated summary
                summary = format_plan_summary(plan_data)
                
                # Check what's still missing
                all_fields = [
                    'plan_name', 'plan_period', 'territory', 'quota', 
                    'commission_structure', 'tiers', 'bonus_rules', 
                    'sales_target', 'effective_dates'
                ]
                still_missing = [f for f in all_fields if not plan_data.get(f)]
                
                logger.info(f"[WIZARD] Still missing fields: {still_missing}")
                
                if not still_missing or len(still_missing) <= 2:
                    return {
                        "response": f"Perfect! Here's your complete plan:\n\n{summary}\n\n✅ **Ready to submit!**\nType 'submit' to save this plan, or 'edit' to make changes."
                    }
                else:
                    missing_str = ', '.join([f.replace('_', ' ') for f in still_missing[:3]])
                    return {
                        "response": f"Great! I've updated your plan:\n\n{summary}\n\nℹ️ Still missing: {missing_str}\n\nYou can provide these details, type 'submit' to save with current info, or 'edit' to change values."
                    }
        
        # Fallback for no plan data
        return {
            "response": "No active plan creation session. Please type 'create plan' to start creating a new commission plan."
        }
    
    except Exception as e:
        logger.error(f"[WIZARD] Critical error in handle_message_in_flow: {e}", exc_info=True)
        return {
            "response": f"Sorry, there was an error processing your request. Please try again or type 'clear session' to start over."
        }

'''def format_plan_summary(plan_data):
    """Format plan data into a Markdown summary for ReactMarkdown"""
    lines = []
    
    # Header (added once at the start)
    lines.append("Perfect! Here's your complete plan:")
    lines.append("")  # Blank line
    lines.append("## 📋 PLAN SUMMARY")
    lines.append("")  # Blank line after header
    
    field_labels = {
        'plan_name': '📝 Plan Name',
        'plan_period': '📅 Period',
        'territory': '🌍 Territory',
        'quota': '🎯 Quota',
        'commission_structure': '💰 Commission Structure',
        'tiers': '📊 Commission Tiers',
        'bonus_rules': '🎁 Bonus Rules',
        'sales_target': '📈 Sales Target',
        'effective_dates': '📆 Effective Dates'
    }
    
    for field, label in field_labels.items():
        value = plan_data.get(field, 'Not specified')
        
        # Special handling for commission tiers
        if field == 'tiers' and isinstance(value, list) and value:
            lines.append(f"**{label}:**")
            for tier in value:
                if isinstance(tier, dict):
                    threshold = tier.get('threshold', tier.get('min', 'Unknown'))
                    rate = tier.get('rate', tier.get('commission', 'Unknown'))
                    # Ensure rate has % sign
                    rate_str = str(rate)
                    if not rate_str.endswith('%'):
                        rate_str = rate_str + '%'
                    lines.append(f"- {threshold}: {rate_str}")
            lines.append("")  # Blank line after bullet list
        else:
            # Special handling for effective dates
            if field == 'effective_dates' and isinstance(value, dict):
                start = value.get('start', 'Not set')
                end = value.get('end', 'Not set')
                value = f"{start} to {end}"
            elif isinstance(value, list):
                value = ', '.join(str(v) for v in value) if value else 'Not specified'
            elif not value:
                value = 'Not specified'
            
            lines.append(f"**{label}:** {value}")
            lines.append("")  # Blank line after each field
    
    # Submit instructions (added once at the end)
    lines.append("✅ Ready to submit!")
    lines.append("Type 'submit' to save this plan, or 'edit' to make changes.")
    
    # Join with newlines and return
    return "\n".join(lines)'''

def format_plan_summary(plan_data, show_all_fields=False):
    """
    Format plan data into a Markdown summary for ReactMarkdown
    
    Args:
        plan_data: Dictionary containing plan information
        show_all_fields: If True, show complete API payload with all defaults
    """
    lines = []
    
    if not show_all_fields:
        # NORMAL SUMMARY (during editing)
        lines.append("Perfect! Here's your complete plan:")
        lines.append("")
        lines.append("## 📋 PLAN SUMMARY")
        lines.append("")
        
        field_labels = {
            'plan_name': '📝 Plan Name',
            'plan_period': '📅 Period',
            'territory': '🌍 Territory',
            'quota': '🎯 Quota',
            'commission_structure': '💰 Commission Structure',
            'tiers': '📊 Commission Tiers',
            'bonus_rules': '🎁 Bonus Rules',
            'sales_target': '📈 Sales Target',
            'effective_dates': '📆 Effective Dates'
        }
        
        for field, label in field_labels.items():
            value = plan_data.get(field, None)
            
            is_empty = (
                value is None or 
                value == '' or 
                value == 'Not specified' or
                (isinstance(value, list) and len(value) == 0) or
                (isinstance(value, dict) and not any(value.values()))
            )
            
            if is_empty:
                continue
            
            if field == 'tiers':
                if isinstance(value, list) and value:
                    lines.append(f"**{label}:**")
                    for tier in value:
                        if isinstance(tier, dict):
                            threshold = tier.get('threshold', tier.get('min', 'Unknown'))
                            rate = tier.get('rate', tier.get('commission', 'Unknown'))
                            rate_str = str(rate)
                            if not rate_str.endswith('%'):
                                rate_str = rate_str + '%'
                            lines.append(f"- {threshold}: {rate_str}")
                    lines.append("")
            else:
                if field == 'effective_dates':
                    if isinstance(value, dict) and any(value.values()):
                        start = value.get('start', 'Not set')
                        end = value.get('end', 'Not set')
                        value = f"{start} to {end}"
                    else:
                        value = '*Not specified*'
                elif isinstance(value, list):
                    value = ', '.join(str(v) for v in value) if value else '*Not specified*'
                elif not value or value == 'Not specified':
                    value = '*Not specified*'
                
                lines.append(f"**{label}:** {value}")
                lines.append("")
        
        lines.append("✅ Ready to submit!")
        lines.append("Type 'submit' to see full details and save this plan, or 'edit' to make changes.")
    
    else:
        # COMPLETE API PAYLOAD SUMMARY (before final save)
        # Build the actual payload that will be sent
        payload = build_simple_payload_from_plan_data(plan_data)
        
        lines.append("⚠️ **FINAL CONFIRMATION - Complete Plan Details**")
        lines.append("")
        lines.append("This shows EVERYTHING that will be saved to the database:")
        lines.append("")
        
        # Top-level program fields
        lines.append("## 📋 PROGRAM DETAILS")
        lines.append("")
        
        program_name = payload.get('ProgramName', '')
        if program_name:
            lines.append(f"**Program Name:** {program_name} *(from user input)*")
        else:
            lines.append(f"**Program Name:** *(not set)*")
        lines.append("")
        
        org_code = payload.get('OrgCode', '')
        if org_code:
            lines.append(f"**Organization Code:** {org_code} *(from your login)*")
        else:
            lines.append(f"**Organization Code:** *(not set)*")
        lines.append("")
        
        lines.append(f"**Program ID:** {payload.get('id', 0)} *(will be auto-assigned)*")
        lines.append("")
        
        client_id = payload.get('client_id', '')
        if client_id:
            lines.append(f"**Client ID:** {client_id} *(from organization)*")
        else:
            lines.append(f"**Client ID:** *(not set)*")
        lines.append("")
        
        org_id = payload.get('org_id', '')
        if org_id:
            lines.append(f"**Organization ID:** {org_id} *(from your login)*")
        else:
            lines.append(f"**Organization ID:** *(not set)*")
        lines.append("")
        
        # Dates
        lines.append("## 📅 VALIDITY PERIOD")
        lines.append("")
        
        valid_from = payload.get('ValidFrom', '')
        valid_to = payload.get('ValidTo', '')
        
        if valid_from:
            lines.append(f"**Valid From:** {valid_from} *(from user input)*")
        else:
            lines.append(f"**Valid From:** *(not set)*")
        lines.append("")
        
        if valid_to:
            lines.append(f"**Valid To:** {valid_to} *(from user input)*")
        else:
            lines.append(f"**Valid To:** *(not set)*")
        lines.append("")
        
        # Schedules
        lines.append("## 🕐 SCHEDULES")
        lines.append("")
        
        payment_schedule = payload.get('PaymentSchedule', '')
        payment_label = payload.get('PaymentScheduleLabel', '')
        lines.append(f"**Payment Schedule:** {payment_schedule} - {payment_label} *(default)*")
        lines.append("")
        
        calc_schedule = payload.get('CalculationSchedule', '')
        calc_label = payload.get('CalculationScheduleLabel', '')
        lines.append(f"**Calculation Schedule:** {calc_schedule} - {calc_label} *(default)*")
        lines.append("")
        
        # Assignment
        lines.append("## 👤 ASSIGNMENT")
        lines.append("")
        
        assignee_name = payload.get('AssigneeName', '')
        if assignee_name:
            lines.append(f"**Assignee Name:** {assignee_name} *(from user input or default)*")
        else:
            lines.append(f"**Assignee Name:** *(not set)*")
        lines.append("")
        
        lines.append(f"**Assignee ID:** {payload.get('AssigneeId', '')} *(default)*")
        lines.append("")
        
        lines.append(f"**Object Type:** {payload.get('ObjectType', '')} *(default)*")
        lines.append("")
        
        lines.append(f"**Table Name:** {payload.get('TableName', '')} *(default)*")
        lines.append("")
        
        lines.append(f"**Table ID:** {payload.get('TableId', '')} *(default)*")
        lines.append("")
        
        # Review status
        lines.append("## 📝 REVIEW STATUS")
        lines.append("")
        
        reviewed_by = payload.get('ReviewedBy', '')
        if reviewed_by:
            lines.append(f"**Reviewed By:** {reviewed_by}")
        else:
            lines.append(f"**Reviewed By:** *(empty - not yet reviewed)*")
        lines.append("")
        
        review_status = payload.get('ReviewStatus', '')
        if review_status:
            lines.append(f"**Review Status:** {review_status}")
        else:
            lines.append(f"**Review Status:** *(empty - pending)*")
        lines.append("")
        
        # Plan conditions
        plan_conditions = payload.get('PlanConditionsDto', [])
        if plan_conditions:
            lines.append("## 📐 PLAN RULES & CONDITIONS")
            lines.append("")
            
            for idx, condition in enumerate(plan_conditions, 1):
                if len(plan_conditions) > 1:
                    lines.append(f"### Rule #{idx}")
                    lines.append("")
                
                lines.append(f"**Plan Type:** {condition.get('PlanType', '')} *(default)*")
                lines.append("")
                
                lines.append(f"**Plan Type Master ID:** {condition.get('PlanTypeMasterId', '')} *(default)*")
                lines.append("")
                
                lines.append(f"**Sequence:** {condition.get('Sequence', '')} *(default)*")
                lines.append("")
                
                lines.append(f"**Step:** {condition.get('Step', '')} *(default)*")
                lines.append("")
                
                plan_desc = condition.get('PlanDescription', '')
                if plan_desc:
                    lines.append(f"**Plan Description:** {plan_desc}")
                else:
                    lines.append(f"**Plan Description:** *(not set)*")
                lines.append("")
                
                plan_params = condition.get('PlanParamsJson', '')
                if plan_params and plan_params != "[]":
                    lines.append(f"**Plan Parameters:** {plan_params}")
                else:
                    lines.append(f"**Plan Parameters:** *(empty array)*")
                lines.append("")
                
                # Structure details
                tiered = condition.get('Tiered', False)
                lines.append(f"**Tiered:** {'Yes' if tiered else 'No'} *(from user input)*")
                lines.append("")
                
                lines.append(f"**Category Type:** {condition.get('CategoryType', '')} *(from user input)*")
                lines.append("")
                
                lines.append(f"**Range Type:** {condition.get('RangeType', '')} *(default)*")
                lines.append("")
                
                lines.append(f"**Value Type:** {condition.get('ValueType', '')} *(default)*")
                lines.append("")
                
                lines.append(f"**Plan Base:** {condition.get('PlanBase', '')} *(default)*")
                lines.append("")
                
                lines.append(f"**Base Value:** {condition.get('BaseValue', '')} *(default)*")
                lines.append("")
                
                show_assignment = condition.get('ShowAssignment', False)
                lines.append(f"**Show Assignment:** {'Yes' if show_assignment else 'No'} *(default)*")
                lines.append("")
                
                is_step = condition.get('isStep', False)
                lines.append(f"**Is Step:** {'Yes' if is_step else 'No'} *(default)*")
                lines.append("")
                
                abs_calc = condition.get('AbsoluteCalculation', False)
                lines.append(f"**Absolute Calculation:** {'Yes' if abs_calc else 'No'} *(default)*")
                lines.append("")
                
                # Rule validity dates
                rule_from = condition.get('ValidFrom', '')
                rule_to = condition.get('ValidTo', '')
                
                if rule_from:
                    lines.append(f"**Rule Valid From:** {rule_from}")
                else:
                    lines.append(f"**Rule Valid From:** *(not set)*")
                lines.append("")
                
                if rule_to:
                    lines.append(f"**Rule Valid To:** {rule_to}")
                else:
                    lines.append(f"**Rule Valid To:** *(not set)*")
                lines.append("")
                
                # JSON data (commission tiers)
                json_data = condition.get('JsonData', [])
                if json_data:
                    lines.append("### 💰 Commission Structure")
                    lines.append("")
                    
                    for data_idx, data_item in enumerate(json_data, 1):
                        if len(json_data) > 1:
                            lines.append(f"**Structure #{data_idx}:**")
                            lines.append("")
                        
                        commission = data_item.get('commission', '')
                        if commission:
                            lines.append(f"**Base Commission:** {commission}%")
                            lines.append("")
                        else:
                            lines.append(f"**Base Commission:** *(not set)*")
                            lines.append("")
                        
                        # Commission tiers
                        tiers = data_item.get('tiers', [])
                        if tiers:
                            lines.append("**Commission Tiers:**")
                            for tier in tiers:
                                from_val = tier.get('from_value', '')
                                to_val = tier.get('to_value', '')
                                comm = tier.get('commission', '')
                                
                                if to_val:
                                    lines.append(f"- From {from_val} to {to_val}: {comm}%")
                                else:
                                    lines.append(f"- From {from_val} and above: {comm}%")
                            lines.append("")
                        
                        # Currency
                        currency = data_item.get('currency', '')
                        if currency:
                            lines.append(f"**Currency:** {currency} *(default - USD)*")
                        else:
                            lines.append(f"**Currency:** *(not set)*")
                        lines.append("")
                        
                        # Plan value ranges
                        ranges = data_item.get('PlanValueRangesDto', [])
                        if ranges and len(ranges) > 0:
                            range_item = ranges[0]
                            has_ranges = any([
                                range_item.get('BusinessPartner'),
                                range_item.get('Product'),
                                range_item.get('Territory'),
                                range_item.get('Plant'),
                                range_item.get('Group')
                            ])
                            
                            if has_ranges:
                                lines.append("**Plan Value Ranges:**")
                                if range_item.get('BusinessPartner'):
                                    lines.append(f"- Business Partner: {range_item.get('BusinessPartner')}")
                                if range_item.get('Product'):
                                    lines.append(f"- Product: {range_item.get('Product')}")
                                if range_item.get('Territory'):
                                    lines.append(f"- Territory: {range_item.get('Territory')}")
                                if range_item.get('Plant'):
                                    lines.append(f"- Plant: {range_item.get('Plant')}")
                                if range_item.get('Group'):
                                    lines.append(f"- Group: {range_item.get('Group')}")
                                lines.append("")
                            else:
                                lines.append("**Plan Value Ranges:** *(all empty arrays)*")
                                lines.append("")
                
                # Plan assignments
                assignments = condition.get('PlanAssignments', [])
                if assignments:
                    lines.append("**Plan Assignments:**")
                    lines.append(f"- {len(assignments)} assignment(s)")
                    lines.append("")
                else:
                    lines.append("**Plan Assignments:** *(empty array)*")
                    lines.append("")
        
        # Final instructions
        lines.append("---")
        lines.append("")
        lines.append("✅ **Type 'confirm'** to save this plan with all the above settings")
        lines.append("✏️ **Type 'edit'** to modify user-provided fields")
        lines.append("❌ **Type 'cancel'** to discard this plan")
    
    return "\n".join(lines)


# --- COMPLETELY REWRITTEN: Build payload for POST from session data ---

def build_simple_payload_from_plan_data(plan_data):
    """
    Build API payload matching the exact structure your .NET backend expects
    """
    # Get dates with defaults
    effective_dates = plan_data.get('effective_dates', {})
    if isinstance(effective_dates, dict):
        start_date = effective_dates.get('start', datetime.now().strftime("%Y-%m-%d"))
        end_date = effective_dates.get('end', (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"))
    else:
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
    
    # Format date properly
    def format_datetime(date_str, is_end_date=False):
        if not date_str:
            return ""
        try:
            if 'T' in date_str:
                return date_str
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if is_end_date:
                return dt.strftime("%Y-%m-%dT23:59:59")
            else:
                return dt.strftime("%Y-%m-%dT00:00:00")
        except:
            return date_str
    
    # Build plan conditions with proper structure
    plan_conditions = []
    
    # Create tiers structure using actual quota
    json_data_list = []
    tiers = []
    quota = plan_data.get('quota', 0)
    
    if plan_data.get('tiers'):
        for idx, tier in enumerate(plan_data.get('tiers', [])):
            if isinstance(tier, dict):
                threshold = tier.get('threshold', '').lower()
                rate = tier.get('rate', '5').replace('%', '')
                
                if 'below' in threshold:
                    # Below quota tier
                    tiers.append({
                        "from_value": "0",
                        "to_value": str(quota) if quota else "100000",
                        "commission": rate
                    })
                elif 'above' in threshold or 'over' in threshold:
                    # Above quota tier
                    tiers.append({
                        "from_value": str(quota) if quota else "0",
                        "to_value": "",  # Empty means "and above"
                        "commission": rate
                    })
                else:
                    # Generic tier if threshold not recognized
                    tiers.append({
                        "from_value": str(quota * idx) if quota else "0",
                        "to_value": str(quota * (idx + 1)) if quota else "",
                        "commission": rate
                    })
    else:
        # Default tier based on commission_structure
        commission = plan_data.get('commission_structure', '8%').replace('%', '')
        tiers.append({
            "from_value": "0",
            "to_value": "",
            "commission": commission
        })
    
    '''json_data_item = {
        "commission": "",
        "tiers": tiers,
        "PlanValueRangesDto": [{
            "BusinessPartner": [],
            "Product": [],
            "Territory": [],
            "Plant": [],
            "Group": []
        }],
        "currency": 78  # Default USD
    }
    json_data_list.append(json_data_item)'''

    # FULL metadata stored here (no length limit in JsonData)
    full_metadata = {
        "plan_period": plan_data.get('plan_period', ''),
        "quota": plan_data.get('quota', 0),
        "sales_target": plan_data.get('sales_target', 0),
        "bonus_rules": plan_data.get('bonus_rules', ''),
        "commission_structure": plan_data.get('commission_structure', ''),
        "territory": plan_data.get('territory', ''),
        "original_tiers": plan_data.get('tiers', [])
    }

    json_data_item = {
        "commission": "",
        "tiers": tiers,
        "plan_metadata": full_metadata,  # ✅ ALL plan data saved here
        "PlanValueRangesDto": [{
            "BusinessPartner": [],
            "Product": [],
            "Territory": [],
            "Plant": [],
            "Group": []
        }],
        "currency": 78
    }
    json_data_list.append(json_data_item)

    # Store ALL extracted plan data as metadata
    '''plan_metadata = {
        "plan_period": plan_data.get('plan_period', ''),
        "quota": plan_data.get('quota', 0),
        "sales_target": plan_data.get('sales_target', 0),
        "bonus_rules": plan_data.get('bonus_rules', ''),
        "commission_structure": plan_data.get('commission_structure', ''),
        "territory": plan_data.get('territory', ''),
        "original_tiers": plan_data.get('tiers', [])
    }'''

    # Short summary for PlanParamsJson (VARCHAR 250 constraint)
    short_metadata = {
        "q": plan_data.get('quota', 0),
        "t": plan_data.get('sales_target', 0)
    }

    period = plan_data.get('plan_period', '')
    if period and len(period) < 20:
        short_metadata["p"] = period[:15]

    territory = plan_data.get('territory', '')
    if territory and len(territory) < 20:
        short_metadata["ter"] = territory[:15]

    # Build the rule
    rule = {
        "PlanType": "Commission",
        "PlanTypeMasterId": 1,
        "Sequence": 100,
        "Step": 0,
        "PlanDescription": f"{plan_data.get('plan_name', 'Commission')} Plan"[:250],
        "PlanParamsJson": json.dumps(short_metadata),  # ✅ Short version (under 100 chars)
        "PlanParams": json.dumps(short_metadata),      # ✅ Same short version
        "Tiered": bool(plan_data.get('tiers')),
        "CategoryType": "Tiered" if plan_data.get('tiers') else "Flat",
        "RangeType": "Amount",
        "ValueType": "Percentage",
        "ValidFrom": start_date,
        "ValidTo": end_date,
        "PlanBase": "Revenue",
        "BaseValue": "Net",
        "JsonData": json_data_list,
        "ShowAssignment": True,
        "isStep": False,
        "AbsoluteCalculation": False,
        "PlanAssignments": []
    }
    plan_conditions.append(rule)

    # Get org and client IDs
    org_id = getattr(request, "org_id", None)
    client_id = get_client_id_for_org(org_id) if org_id else None
    
    # Build final payload matching PostProgramCreation structure
    payload = {
        "id": 0,
        "OrgCode": getattr(request, "org_code", ""),
        "PaymentSchedule": 1,  # Default schedule ID
        "ProgramName": plan_data.get('plan_name', 'Untitled Plan'),
        "CalculationSchedule": 1,  # Default schedule ID
        "ValidFrom": format_datetime(start_date, False),
        "ValidTo": format_datetime(end_date, True),
        "ReviewedBy": "",
        "ReviewStatus": "",
        "ObjectType": "Invoices",
        "TableId": 1,  # Default
        "AssigneeId": 1,  # Default
        "TableName": "hierarchy_master_instance",
        "AssigneeName": plan_data.get('territory', 'All'),
        "PlanConditionsDto": plan_conditions,
        "CalculationScheduleLabel": "Default Schedule",
        "PaymentScheduleLabel": "Default Schedule",
        "Id": 0
    }
    
    if org_id:
        payload["org_id"] = org_id
    if client_id:
        payload["client_id"] = client_id
    
    logger.info(f"[build_simple_payload] Generated payload: {json.dumps(payload, indent=2)[:1000]}")
    return payload

@app.route("/chat/history", methods=["GET"])
@verify_jwt_token
def chat_history():
    user_id = request.args.get("user_id")
    client_id = request.args.get("client_id")
    session_id = request.args.get("session_id")

    if not (user_id and session_id):
        return jsonify({"error": "Missing user_id or session_id"}), 400
    
    # Get org_id and derive client_id if not provided
    org_id = getattr(request, "org_id", None)
    
    if not client_id and org_id:
        client_id = get_client_id_for_org(org_id)
    
    if not client_id:
        return jsonify({"error": "Unable to determine client_id"}), 400
    
    history = get_chat_history(client_id, session_id)
    return jsonify({"history": history})

# --- Chat endpoint - UPDATED WITH INTENT RECOGNITION AND RAG INTEGRATION ---
@app.route("/chat", methods=["POST"])
@verify_jwt_token
def chat_endpoint():
    data = request.json or {}
    user_msg = (data.get("message") or "").strip()

    # Force permanent session
    session.permanent = True

    # DEBUG: Log session state at start of request
    logger.info(f"[SESSION-DEBUG] Request start - Session ID: {session.get('_id', 'NO_ID')}")
    logger.info(f"[SESSION-DEBUG] Session mode: {session.get('mode', 'NO_MODE')}")

    # DEBUG: Check current session state
    logger.info(f"[SESSION-DEBUG] Current session mode: {session.get('mode')}")
    logger.info(f"[SESSION-DEBUG] Session keys: {list(session.keys())}")
    
    #user_id = data.get("user_id") or data.get("userId") or None
    # Get user_id from BOTH request body AND URL parameters
    user_id = (
        data.get("user_id") or 
        data.get("userId") or 
        request.args.get("user_id") or 
        request.args.get("userId") or 
        getattr(request, "user_id", None) or  # ← From JWT
        None
    )

    # DEBUG: Log where user_id came from
    if user_id:
        if data.get("user_id") or data.get("userId"):
            logger.info(f"[DEBUG-USER] user_id from request body: {user_id}")
        elif request.args.get("user_id") or request.args.get("userId"):
            logger.info(f"[DEBUG-USER] user_id from URL params: {user_id}")
        else:
            logger.info(f"[DEBUG-USER] user_id from JWT token: {user_id}")
    else:
        logger.error(f"[DEBUG-USER] No user_id found in body, params, or JWT!")

    # If no user_id found, extract from JWT token
    if not user_id:
        user_id = getattr(request, "user_id", None)  # From JWT @verify_jwt_token
        if user_id:
            logger.info(f"[DEBUG-USER] user_id extracted from JWT: {user_id}")
        else:
            logger.error(f"[DEBUG-USER] No user_id found in JWT either!")

    # DEBUG: Log where user_id came from
    if user_id:
        if data.get("user_id") or data.get("userId"):
            logger.info(f"[DEBUG-USER] user_id from request body: {user_id}")
        else:
            logger.info(f"[DEBUG-USER] user_id from URL params: {user_id}")
    else:
        logger.error(f"[DEBUG-USER] No user_id found in body or params!")

    org_id = getattr(request, "org_id", None)
    client_id = get_client_id_for_org(org_id) if org_id else None
    session_id = data.get("session_id") or f"{user_id}_{int(time.time())}"

    logger.info(f"[DEBUG-SESSION] Using session_id: {session_id}")

    # Session state reload (server-side per user)
    if user_id:
        state = load_user_state(user_id) or {}
        logger.info(f"[SESSION-DEBUG] Loaded user state for {user_id}: {list(state.keys())}")
        # session.clear()
        jwt = session.get("jwt_token")
        session.clear()
        if jwt:
            session["jwt_token"] = jwt

        for k, v in state.items():
            session[k] = v
            logger.info(f"[SESSION-DEBUG] Restored session key: {k} = {type(v)}")

        session.modified = True

        # Debug: Log final session state
        logger.info(f"[SESSION-DEBUG] Final session after restore: {list(session.keys())}")
        if session.get("mode"):
            logger.info(f"[SESSION-DEBUG] Session mode after restore: {session.get('mode')}")
            logger.info(f"[SESSION-DEBUG] Plan data exists: {bool(session.get('plan_data'))}")

        # Save user message
        save_message(client_id, session_id, {"sender": "user", "text": user_msg})

    # PRIORITY 1: If already in wizard mode, handle directly (bypass ALL intent detection)
    if session.get("mode") == "plan_creation":
        logger.info("🎯 [WIZARD] User already in plan creation mode - bypassing intent detection")
        resp = handle_message_in_flow(user_msg)
        session.modified = True  # Force session save
        if user_id:
            save_user_state(user_id, dict(session))
        save_message(client_id, session_id, {"sender": "bot", "text": resp.get("response")})
        history = get_chat_history(client_id, session_id)
        return jsonify({"response": resp.get("response"), "history": history})
    


    # Continue plan flow if in progress (wizard or paragraph)
    

    # Handle session clear
    if user_msg.lower() in ["cancel", "clear session", "/clear_session"]:
        # session.clear()
        jwt = session.get("jwt_token")
        session.clear()
        if jwt:
            session["jwt_token"] = jwt

        if user_id:
            clear_user_state(user_id)
        return jsonify({"response": "Session cleared. You can now ask questions or start plan creation."})

    # --- Intent: Plan (Wizard or Paragraph, powered by LLM) ---
    from intent_detection import is_plan_creation_intent
    from intent_detection import extract_plan_fields_claude

    logger.info(f"[DEBUG-INTENT] Checking plan intent for message: {user_msg}")
    plan_intent = is_plan_creation_intent(user_msg, mcp_client.claude_complete)
    logger.info(f"[DEBUG-INTENT] Plan intent for message: {plan_intent}")
    
    if plan_intent:
    # if is_plan_creation_intent(user_msg, mcp_client.claude_complete): 
        #logger.info("🎯 [PLAN] Starting plan creation flow (NLP Plan Builder)")
        logger.info("🎯 [PLAN] Starting NLP Plan Builder flow, raw user message: %s", user_msg)
        # DEBUG: Check user_id availability
        logger.info(f"[DEBUG-SAVE] user_id available for save: {user_id}")
        logger.info(f"[DEBUG-SAVE] user_id type: {type(user_id)}")

        result = extract_plan_struct_from_text(user_msg, mcp_client.claude_complete)
        logger.info("[PLAN] NLP extraction result: %s", result)
        
        # CRITICAL: Set session mode and force session save
        session['mode'] = 'plan_creation'
        session.modified = True
        logger.info(f"[SESSION-DEBUG] Set session mode to plan_creation and marked modified")

        # DEBUG: Check session before save
        logger.info(f"[DEBUG-SAVE] Session before save attempt: {list(session.keys())}")

        # CRITICAL: Save immediately after setting mode (ALWAYS, regardless of NLP result)
        if user_id:
            logger.info(f"[DEBUG-SAVE] Attempting to save state for user_id: {user_id}")
            try:
                save_user_state(user_id, dict(session))
                logger.info(f"[SESSION-DEBUG] Saved user state with keys: {list(dict(session).keys())}")
                logger.info(f"[DEBUG-SAVE] Save successful!")
            except Exception as e:
                logger.error(f"[DEBUG-SAVE] Save failed with error: {e}")
        else:
            logger.error(f"[DEBUG-SAVE] Cannot save - user_id is None/empty: {user_id}")

        if result["status"] == "ok":
            extracted = result["extracted"]
            missing = result["missing_fields"]
            logger.info("[PLAN] Initial extracted fields: %s", extracted)
            logger.info("[PLAN] Missing fields to prompt: %s", missing)
            session["mode"] = "plan_creation"
            session["plan_data"] = extracted
            session.modified = True

            # Save again after adding plan data
            if user_id:
                save_user_state(user_id, dict(session))
                logger.info(f"[SESSION-DEBUG] Saved user state with keys: {list(dict(session).keys())}")

            if missing:
                question = f"Please provide the following missing details for your plan: {', '.join(missing)}"
            else:
                #summary = json.dumps(extracted, indent=2)
                #question = f"Here's a summary of your new plan. Do you want to submit or edit?\n{summary}"
                summary = format_plan_summary(extracted) 
                question = summary
            logger.info("[PLAN] Wizard initial response sent: %s", question)
            save_message(client_id, session_id, {"sender": "bot", "text": question})
            history = get_chat_history(client_id, session_id)
            return jsonify({"response": question, "history": history})
        else:
            logger.error("[PLAN] NLP extraction error: %s", result.get("error"))
            reply = "Sorry, I couldn't extract the plan details due to an error. Please rephrase or try again."
            save_message(client_id, session_id, {"sender": "bot", "text": reply})
            history = get_chat_history(client_id, session_id)
            return jsonify({"response": reply, "history": history})



    # ==================== ML PREDICTION & OPTIMIZATION ====================
    
    if is_ml_prediction_intent(user_msg) or is_ml_optimization_intent(user_msg):
        logger.info("🤖 [ML] Processing ML prediction/optimization request")
        
        org_id = getattr(request, "org_id", None)
        client_id = get_client_id_for_org(org_id) if org_id else None
        
        if not org_id or not client_id:
            ml_response = ("I need your organization context to generate predictions. "
                          "Please make sure you're logged in with valid credentials.")
        else:
            try:
                ml_response = handle_ml_request(user_msg, org_id, client_id)
                
                # Save state
                if user_id:
                    save_user_state(user_id, dict(session))
                
                # Save to Redis
                save_message(client_id, session_id, {"sender": "bot", "text": ml_response})
                history = get_chat_history(client_id, session_id)
                
                return jsonify({"response": ml_response, "history": history})
                
            except Exception as e:
                logger.error(f"❌ [ML] Error: {e}", exc_info=True)
                ml_response = (f"Sorry, I encountered an error while processing your ML request. "
                              f"Please try again or rephrase your question.")
        
        # Return error response
        if user_id:
            save_user_state(user_id, dict(session))
        save_message(client_id, session_id, {"sender": "bot", "text": ml_response})
        history = get_chat_history(client_id, session_id)
        return jsonify({"response": ml_response, "history": history})
    
    # ==================== END ML SECTION ====================

    # --- Database Query ---
    if is_database_query_intent(user_msg):
        logger.info("🗄️ [DB] Processing database query")

        # Identify pagination-triggering user queries
        UM = user_msg.lower().strip()
        if "show all plans" in UM:
            reset_plan_offset()
        elif "show more" in UM:
            increment_plan_offset(10)  # or your preferred page size

        offset = get_plan_offset()
        logger.info(f"[PAGINATION] Current plan offset: {offset}")
        try:
            db_response = handle_database_query(user_msg, offset=offset)
            if user_id:
                save_user_state(user_id, dict(session))
            save_message(client_id, session_id, {"sender": "bot", "text": db_response})
            history = get_chat_history(client_id, session_id)
            return jsonify({"response": db_response, "history": history})
        except Exception as e:
            logger.error(f"❌ [DB] Error in database processing: {e}")
            return jsonify({"response": f"Sorry, error: {e}"})


    # --- RAG fallback ---
    logger.info("🧠 [RAG] Processing general query")
    try:
        rag_response = handle_rag_query(user_msg)
        if user_id:
            save_user_state(user_id, dict(session))
        save_message(client_id, session_id, {"sender": "bot", "text": rag_response})
        history = get_chat_history(client_id, session_id)
        return jsonify({"response": rag_response, "history": history})
    except Exception as e:
        logger.error(f"❌ [RAG] Error in RAG processing: {e}")
        error_text = str(e)
        if "429" in error_text or "rate_limit_error" in error_text or "Too Many Requests" in error_text:
            return jsonify({"response": "The system is currently handling too many AI requests. Please wait a few seconds and try again."})
        if user_id:
            save_user_state(user_id, dict(session))
        return jsonify({"response": f"Sorry, error: {e}"})
    
# --- Home route ---
@app.route("/", methods=["GET"])
def home():
    try:
        return render_template("index.html")
    except Exception:
        # fallback to serving static index.html if templates not present
        root = os.path.dirname(__file__)
        return send_from_directory(root, "index.html")

# --- Debug route to inspect session ---
@app.route("/_debug_session", methods=["GET"])
def debug_session():
    user_id = request.args.get("user_id")
    out = {"flask_session": {k: session.get(k) for k in session.keys()}}
    if user_id:
        out["user_state"] = load_user_state(user_id)
    return jsonify(out)

# --- API connectivity test route ---
@app.route("/_test_api", methods=["GET"])
def test_api_connectivity():
    """Test endpoint to check API connectivity"""
    try:
        # Test basic connectivity
        test_url = f"{API_BASE_URL}/api/PlanCreationWebApi/GetCommissionSchedule"
        r = requests.get(test_url, headers=api_headers(), timeout=5, verify=False)
        
        return jsonify({
            "api_base_url": API_BASE_URL,
            "test_endpoint_status": r.status_code,
            "jwt_present": bool(JWT_TOKEN),
            "test_response": r.text[:500] if r.status_code == 200 else "Failed"
        })
    except Exception as e:
        return jsonify({"error": str(e), "api_base_url": API_BASE_URL})

@app.route("/_test_db", methods=["GET"])
def test_db_connectivity():
    """Test endpoint to check database connectivity"""
    try:
        if not mcp_client.is_enabled():
            return jsonify({"error": "Database queries are disabled"})
        
        # Test schema loading
        schema = mcp_client.get_database_schema()
        
        if schema:
            table_count = len(schema.keys())
            table_names = list(schema.keys())[:5]  # First 5 tables
            
            return jsonify({
                "database_host": os.getenv('DB_HOST', 'not_set'),
                "database_name": os.getenv('DB_NAME', 'not_set'),
                "schema_loaded": True,
                "table_count": table_count,
                "sample_tables": table_names,
                "status": "success"
            })
        else:
            return jsonify({"error": "Failed to load database schema"})
            
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})
    
# --- Clear vector cache endpoint ---
@app.route("/_clear_cache", methods=["POST"])
def clear_cache_endpoint():
    """Endpoint to clear the RAG vector cache"""
    try:
        clear_vector_cache()
        return jsonify({"message": "Vector cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    body = request.get_json()
    username = body.get('username')
    password = body.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    # Call the .NET API
    r = requests.post(
        "https://commissions.callippus.co.uk/api/authenticate",
        json={'username': username, 'password': password}
    )
    if r.status_code != 200:
        return jsonify({'error': 'Invalid credentials'}), 401
    data = r.json()
    token = data.get('id_token') or data.get('token')
    if not token:
        return jsonify({'error': 'No JWT returned'}), 500

    # Store in session
    session['jwt_token'] = token
    print("[DEBUG] jwt_token after setting in session:", session.get('jwt_token'))
    return jsonify({'success': True})

# ADD THESE NEW ENDPOINTS before if __name__ == "__main__":

@app.route("/api/ml/predict", methods=["POST"])
@verify_jwt_token
def api_ml_predict():
    """
    Direct API endpoint for ML predictions
    
    Body: {
        "percentage_change": 10.0,
        "plan_id": 123  // optional
    }
    """
    try:
        data = request.json or {}
        percentage = data.get('percentage_change')
        plan_id = data.get('plan_id')
        
        if percentage is None:
            return jsonify({
                'success': False,
                'error': 'percentage_change is required'
            }), 400
        
        org_id = getattr(request, "org_id", None)
        client_id = get_client_id_for_org(org_id) if org_id else None
        
        if not org_id or not client_id:
            return jsonify({
                'success': False,
                'error': 'Organization context required'
            }), 401
        
        from ml_advanced_predictor import tenant_predictor
        
        result = tenant_predictor.predict_commission_impact(
            org_id=org_id,
            client_id=client_id,
            percentage_change=float(percentage),
            plan_id=plan_id,
            num_simulations=1000
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"ML prediction API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/api/ml/optimize", methods=["POST"])
@verify_jwt_token
def api_ml_optimize():
    """
    Direct API endpoint for plan optimization
    
    Body: {
        "plan_id": 123  // optional
    }
    """
    try:
        data = request.json or {}
        plan_id = data.get('plan_id')
        
        org_id = getattr(request, "org_id", None)
        client_id = get_client_id_for_org(org_id) if org_id else None
        
        if not org_id or not client_id:
            return jsonify({
                'success': False,
                'error': 'Organization context required'
            }), 401
        
        from ml_advanced_predictor import tenant_predictor
        
        result = tenant_predictor.recommend_plan_optimizations(
            org_id=org_id,
            client_id=client_id,
            plan_id=plan_id
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"ML optimization API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/api/ml/apply-change", methods=["POST"])
@verify_jwt_token
def api_ml_apply_change():
    """
    Apply an ML recommendation to a plan
    
    Body: {
        "plan_id": 123,
        "edit_action": { ... }  // from ML recommendation
    }
    """
    try:
        data = request.json or {}
        plan_id = data.get('plan_id')
        edit_action = data.get('edit_action')
        
        if not plan_id or not edit_action:
            return jsonify({
                'success': False,
                'error': 'plan_id and edit_action are required'
            }), 400
        
        org_id = getattr(request, "org_id", None)
        client_id = get_client_id_for_org(org_id) if org_id else None
        
        if not org_id or not client_id:
            return jsonify({
                'success': False,
                'error': 'Organization context required'
            }), 401
        
        result = plan_editor.apply_optimization(
            org_id=org_id,
            client_id=client_id,
            plan_id=plan_id,
            edit_action=edit_action
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"ML apply change API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# --- Run server ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"🚀 Flask running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
