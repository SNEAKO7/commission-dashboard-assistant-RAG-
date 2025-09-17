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


# Import RAG functionality
from rag import retrieve_context, clear_vector_cache

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
app.config["SESSION_PERMANENT"] = False
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
        database=os.getenv("DB_NAME", "commissions_database"),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASSWORD", "pass"),
        port=int(os.getenv("DB_PORT", 321)),
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
            payload = jwt.decode(token, "S_E_C_R_E_T_K_E_Y_Commissions_app", algorithms=['HS256'], audience="http://...", issuer="http://...")
            print("Decoded JWT payload:", payload)
            
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
def is_plan_creation_intent(message: str) -> bool:
    """
    Determine if the user wants to create a commission plan or general query
    """
    message_lower = message.lower().strip()
    
    # Direct plan creation keywords
    plan_keywords = [
        "create plan", "create a plan", "start plan", "start creating plan",
        "new plan", "make plan", "build plan", "commission plan",
        "create commission", "plan creation", "add plan", "setup plan"
    ]
    
    # Check for direct plan creation intent
    for keyword in plan_keywords:
        if keyword in message_lower:
            logger.info(f"🎯 Plan creation intent detected: '{keyword}' found in message")
            return True
    
    logger.info("🤔 No plan creation intent detected - will use RAG")
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
            if not is_plan_creation_intent(message):
                logger.info(f"🎯 Database query intent detected: '{keyword}' found in message")
                return True
    
    return False

def is_database_enabled():
    try:
        from improved_mcp_client import execute_query
        return True
    except ImportError:
        return False

def handle_database_query(user_message: str) -> str:
    """
    Handle database queries using improved MCP client 
    """
    try:
        # Execute the natural language query using the improved MCP/NL→SQL flow
        org_id = getattr(request, 'org_id', None)
        org_code = getattr(request, 'org_code', None)
        client_id = get_client_id_for_org(org_id) if org_id else None
        result = execute_query(user_message, org_id=org_id, client_id=client_id)

        # Format for pretty printing, safe for end user
        formatted_response = format_query_response(result)
        return formatted_response

    except Exception as e:
        logger.error(f"❌ [DB] Error processing database query: {e}")
        return f"Sorry, I encountered an error while querying the database: {str(e)}"


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
        return f"Sorry, I encountered an error while searching the documents: {str(e)}"

# --- HTTP helper ---
def api_headers():
    h = {"Accept": "application/json"}
    token = session.get("jwt_token")
    if JWT_TOKEN:
        h["Authorization"] = f"Bearer {JWT_TOKEN}"
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
        logger.info("[post_program_creation] posting to URL: " + url)
        logger.info("[post_program_creation] payload keys: " + str(list(plan_payload.keys())))
        logger.info("[post_program_creation] posting payload (truncated): " + json.dumps(plan_payload)[:1500])
        
        headers = api_headers()
        headers["Content-Type"] = "application/json"  # Ensure content type is set
        
        r = requests.post(url, json=plan_payload, headers=headers, timeout=30, verify=False)
        logger.info(f"[post_program_creation] status: {r.status_code}")
        logger.info(f"[post_program_creation] response headers: {dict(r.headers)}")
        logger.info(f"[post_program_creation] response text: {r.text[:1000]}")
        
        if r.status_code == 404:
            logger.error("[post_program_creation] 404 error - endpoint not found. Check API_BASE_URL and endpoint path")
            return {"error": "API endpoint not found (404). Check server configuration.", "status_code": 404}
        elif r.status_code == 401:
            logger.error("[post_program_creation] 401 error - authentication failed")
            return {"error": "Authentication failed (401). Check JWT token.", "status_code": 401}
        elif r.status_code == 500:
            logger.error("[post_program_creation] 500 error - server error")
            return {"error": "Server error (500). Check payload format.", "status_code": 500}
        elif r.status_code not in [200, 201]:
            logger.warning(f"[post_program_creation] unexpected status: {r.status_code}")
            return {"error": f"Unexpected response status: {r.status_code}", "status_code": r.status_code, "response": r.text}
        
        try:
            return r.json()
        except:
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
    
    session.clear()
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
    session["org_id"]   = getattr(request, "org_id", None)
    session["org_code"] = getattr(request, "org_code", None)
    session["user_id"]  = getattr(request, "user_id", None)
    session["username"] = getattr(request, "username", None)
    # (client_id/client_code only if needed and correctly mapped)

    session.modified = True

# --- Flow handler (core) - COMPLETE PLAN CREATION LOGIC ---
def handle_message_in_flow(msg):
    """
    msg: string user input
    session keys used:
    - stage: current stage string
    - options: master options dict
    - plan_data, current_rule, current_assignment
    
    Returns: response dict {"response": text}
    """
    stage = session.get("stage")
    opts = session.get("options", {})
    pd = session.get("plan_data", {})
    
    # Helper: send numbered options and store last_options in session for resolving select
    def ask_options(key_name, items, heading):
        # items may be list of strings or dicts
        session["last_options"] = items
        session["last_option_key"] = key_name
        session.modified = True
        return {"response": present_numbered_list(items, heading)}
    
    # Resolve selection using last_options
    def resolve_input(user_text):
        opts_list = session.get("last_options", [])
        resolved = resolve_choice(user_text, opts_list)
        return resolved
    
    m = (msg or "").strip()
    
    # ---------- Phase 1 ----------
    if stage == "plan_name":
        if not m:
            return {"response": "Enter Plan Name:"}
        
        pd["plan_name"] = m
        session["plan_data"] = pd
        session["stage"] = "calculation_schedule"
        session.modified = True
        
        # show calculation schedules
        scheds = opts.get("schedules", [])
        labels = [option_label(s) for s in scheds]
        return ask_options("calculation_schedule", labels, "Choose Calculation Schedule:")
    
    if stage == "calculation_schedule":
        if not m:
            scheds = opts.get("schedules", [])
            labels = [option_label(s) for s in scheds]
            return ask_options("calculation_schedule", labels, "Choose Calculation Schedule:")
        
        sel = resolve_input(m) or m
        
        # map back to schedulerName if possible
        scheds = opts.get("schedules", [])
        # find matching object by label
        chosen_label = None
        for s in scheds:
            lbl = option_label(s)
            if lbl.lower() == str(sel).strip().lower():
                chosen_label = lbl
                break
        
        if chosen_label is None and sel in [option_label(s) for s in scheds]:
            chosen_label = sel
        
        if chosen_label is None:
            # invalid
            labels = [option_label(s) for s in scheds]
            return {"response": present_numbered_list(labels, "Choose Calculation Schedule:") + "\n\nInvalid selection. Choose a number or exact schedule name."}
        
        pd["calculation_schedule"] = chosen_label
        session["plan_data"] = pd
        session["stage"] = "payment_schedule"
        session.modified = True
        
        # show payment schedules (same list)
        labels = [option_label(s) for s in (opts.get("schedules") or [])]
        return ask_options("payment_schedule", labels, "Choose Payment Schedule:")
    
    if stage == "payment_schedule":
        if not m:
            labels = [option_label(s) for s in (opts.get("schedules") or [])]
            return ask_options("payment_schedule", labels, "Choose Payment Schedule:")
        
        sel = resolve_input(m) or m
        labels = [option_label(s) for s in (opts.get("schedules") or [])]
        
        if sel not in labels:
            return {"response": present_numbered_list(labels, "Choose Payment Schedule:") + "\n\nInvalid selection."}
        
        pd["payment_schedule"] = sel
        session["plan_data"] = pd
        session["stage"] = "assignee"
        session.modified = True
        
        # present assignee names
        assignees = opts.get("assignees") or []
        return ask_options("assignee", assignees, "Choose Assignee Name:")
    
    if stage == "assignee":
        if not m:
            assignees = opts.get("assignees") or []
            return ask_options("assignee", assignees, "Choose Assignee Name:")
        
        sel = resolve_input(m) or m
        assignees = opts.get("assignees") or []
        
        if sel not in assignees:
            return {"response": present_numbered_list(assignees, "Choose Assignee Name:") + "\n\nInvalid selection."}
        
        pd["assignee_name"] = sel
        session["plan_data"] = pd
        session["stage"] = "object_type"
        session.modified = True
        
        return ask_options("object_type", opts.get("object_types", OBJECT_TYPES), "Choose Object Type:")
    
    if stage == "object_type":
        if not m:
            return ask_options("object_type", opts.get("object_types", OBJECT_TYPES), "Choose Object Type:")
        
        sel = resolve_input(m) or m
        obj_types = opts.get("object_types", OBJECT_TYPES)
        
        if sel not in obj_types:
            return {"response": present_numbered_list(obj_types, "Choose Object Type:") + "\n\nInvalid selection."}
        
        pd["object_type"] = sel
        session["plan_data"] = pd
        session["stage"] = "valid_from"
        session.modified = True
        
        return {"response": "Enter Valid From date (YYYY-MM-DD):"}
    
    if stage == "valid_from":
        if not m:
            return {"response": "Enter Valid From date (YYYY-MM-DD):"}
        
        try:
            datetime.strptime(m, "%Y-%m-%d")
        except:
            return {"response": "Invalid date format. Use YYYY-MM-DD."}
        
        pd["valid_from"] = m
        session["plan_data"] = pd
        session["stage"] = "valid_to"
        session.modified = True
        
        return {"response": "Enter Valid To date (YYYY-MM-DD):"}
    
    if stage == "valid_to":
        if not m:
            return {"response": "Enter Valid To date (YYYY-MM-DD):"}
        
        try:
            datetime.strptime(m, "%Y-%m-%d")
        except:
            return {"response": "Invalid date format. Use YYYY-MM-DD."}
        
        pd["valid_to"] = m
        session["plan_data"] = pd
        
        # move to Plan Rules
        session["stage"] = "rule_plan_type"
        session.modified = True
        
        # present plan types
        plan_types = opts.get("plan_types") or []
        return ask_options("rule_plan_type", plan_types, "Now entering Plan Rules phase.\nChoose Plan Type for the first rule:")
    
    # ---------- Phase 2 (Plan Rules) ----------
    if stage == "rule_plan_type":
        if not m:
            return ask_options("rule_plan_type", opts.get("plan_types") or [], "Choose Plan Type:")
        
        sel = resolve_input(m) or m
        plan_types = opts.get("plan_types") or []
        
        if sel not in plan_types:
            return {"response": present_numbered_list(plan_types, "Choose Plan Type:") + "\n\nInvalid selection."}
        
        # init new current_rule
        cur = {
            "plan_type": sel,
            "plan_params": [],
            "category_type": None,
            "range_type": None,
            "base_value": None,
            "plan_base": None,
            "value_type": None,
            "valid_from": None,
            "valid_to": None,
            "absolute_calculation": False,
            "sequence": 100,
            "step": 0,
            "assignments": []
        }
        
        session["current_rule"] = cur
        session["stage"] = "rule_plan_params"
        session.modified = True
        
        # show plan params automatically
        params = opts.get("plan_params") or []
        return ask_options("rule_plan_params", params, "Select Plan Params for this rule (choose numbers separated by comma or type names separated by comma):")
    
    if stage == "rule_plan_params":
        params_available = opts.get("plan_params") or []
        
        if not m:
            return ask_options("rule_plan_params", params_available, "Select Plan Params for this rule:")
        
        # user may send comma-separated numbers or names
        parts = [p.strip() for p in m.split(",") if p.strip()]
        chosen = []
        
        for p in parts:
            res = resolve_choice(p, params_available)
            if res:
                chosen.append(res)
            else:
                # if user typed number incorrectly, fallback to matching raw string
                if p in params_available:
                    chosen.append(p)
        
        if not chosen:
            return {"response": present_numbered_list(params_available, "Select Plan Params:") + "\n\nInvalid selection. Choose 1,2 or names separated by comma."}
        
        cur = session.get("current_rule", {})
        cur["plan_params"] = chosen
        session["current_rule"] = cur
        session["stage"] = "rule_category_type"
        session.modified = True
        
        return ask_options("rule_category_type", opts.get("category_types", CATEGORY_TYPES), "Choose Category Type:")
    
    if stage == "rule_category_type":
        if not m:
            return ask_options("rule_category_type", opts.get("category_types", CATEGORY_TYPES), "Choose Category Type:")
        
        sel = resolve_input(m) or m
        cats = opts.get("category_types", CATEGORY_TYPES)
        
        if sel not in cats:
            return {"response": present_numbered_list(cats, "Choose Category Type:") + "\n\nInvalid selection."}
        
        cur = session.get("current_rule")
        cur["category_type"] = sel
        session["current_rule"] = cur
        session["stage"] = "rule_range_type"
        session.modified = True
        
        return ask_options("rule_range_type", opts.get("range_types", RANGE_TYPES), "Choose Range Type:")
    
    if stage == "rule_range_type":
        if not m:
            return ask_options("rule_range_type", opts.get("range_types", RANGE_TYPES), "Choose Range Type:")
        
        sel = resolve_input(m) or m
        rlist = opts.get("range_types", RANGE_TYPES)
        
        if sel not in rlist:
            return {"response": present_numbered_list(rlist, "Choose Range Type:") + "\n\nInvalid selection."}
        
        cur = session.get("current_rule")
        cur["range_type"] = sel
        session["current_rule"] = cur
        session["stage"] = "rule_base_value"
        session.modified = True
        
        return ask_options("rule_base_value", opts.get("base_values", BASE_VALUES), "Choose Base Value:")
    
    if stage == "rule_base_value":
        if not m:
            return ask_options("rule_base_value", opts.get("base_values", BASE_VALUES), "Choose Base Value:")
        
        sel = resolve_input(m) or m
        bvals = opts.get("base_values", BASE_VALUES)
        
        if sel not in bvals:
            return {"response": present_numbered_list(bvals, "Choose Base Value:") + "\n\nInvalid selection."}
        
        cur = session.get("current_rule")
        cur["base_value"] = sel
        session["current_rule"] = cur
        session["stage"] = "rule_plan_base"
        session.modified = True
        
        return ask_options("rule_plan_base", opts.get("plan_bases", PLAN_BASES), "Choose Plan Base:")
    
    if stage == "rule_plan_base":
        if not m:
            return ask_options("rule_plan_base", opts.get("plan_bases", PLAN_BASES), "Choose Plan Base:")
        
        sel = resolve_input(m) or m
        pbs = opts.get("plan_bases", PLAN_BASES)
        
        if sel not in pbs:
            return {"response": present_numbered_list(pbs, "Choose Plan Base:") + "\n\nInvalid selection."}
        
        cur = session.get("current_rule")
        cur["plan_base"] = sel
        session["current_rule"] = cur
        session["stage"] = "rule_value_type"
        session.modified = True
        
        return ask_options("rule_value_type", opts.get("value_types", VALUE_TYPES), "Choose Value Type:")
    
    if stage == "rule_value_type":
        if not m:
            return ask_options("rule_value_type", opts.get("value_types", VALUE_TYPES), "Choose Value Type:")
        
        sel = resolve_input(m) or m
        vts = opts.get("value_types", VALUE_TYPES)
        
        if sel not in vts:
            return {"response": present_numbered_list(vts, "Choose Value Type:") + "\n\nInvalid selection."}
        
        cur = session.get("current_rule")
        cur["value_type"] = sel
        session["current_rule"] = cur
        session["stage"] = "rule_valid_from"
        session.modified = True
        
        return {"response": "Enter Rule Valid From date (YYYY-MM-DD):"}
    
    if stage == "rule_valid_from":
        if not m:
            return {"response": "Enter Rule Valid From date (YYYY-MM-DD):"}
        
        try:
            datetime.strptime(m, "%Y-%m-%d")
        except:
            return {"response": "Invalid date format. Use YYYY-MM-DD."}
        
        cur = session.get("current_rule")
        cur["valid_from"] = m
        session["current_rule"] = cur
        session["stage"] = "rule_valid_to"
        session.modified = True
        
        return {"response": "Enter Rule Valid To date (YYYY-MM-DD):"}
    
    if stage == "rule_valid_to":
        if not m:
            return {"response": "Enter Rule Valid To date (YYYY-MM-DD):"}
        
        try:
            datetime.strptime(m, "%Y-%m-%d")
        except:
            return {"response": "Invalid date format. Use YYYY-MM-DD."}
        
        cur = session.get("current_rule")
        cur["valid_to"] = m
        session["current_rule"] = cur
        session["stage"] = "rule_absolute_calc"
        session.modified = True
        
        return ask_options("rule_absolute_calc", ON_OFF, "Absolute Calculation (On/Off):")
    
    if stage == "rule_absolute_calc":
        if not m:
            return ask_options("rule_absolute_calc", ON_OFF, "Absolute Calculation (On/Off):")
        
        sel = resolve_input(m) or m
        
        if sel not in ON_OFF:
            return {"response": present_numbered_list(ON_OFF, "Absolute Calculation (On/Off):") + "\n\nInvalid selection."}
        
        cur = session.get("current_rule")
        cur["absolute_calculation"] = True if sel.lower() == "on" else False
        session["current_rule"] = cur
        session["stage"] = "rule_add_condition"
        session.modified = True
        
        return ask_options("rule_add_condition", YES_NO, "Add Condition?")
    
    # FIXED: Add Condition Logic
    if stage == "rule_add_condition":
        if not m:
            return ask_options("rule_add_condition", YES_NO, "Add Condition?")
        
        sel = resolve_input(m) or m
        if sel not in YES_NO:
            return {"response": present_numbered_list(YES_NO, "Add Condition?") + "\n\nInvalid selection."}
        
        if sel.lower() == "yes":
            # Save current_rule as a condition
            cur = session.get("current_rule")
            prules = pd.get("plan_rules", [])
            prules.append(dict(cur))
            pd["plan_rules"] = prules
            session["plan_data"] = pd
            
            # Create new condition with increased sequence (backend only)
            new_seq = cur.get("sequence", 100) + 100
            new_condition = {
                "plan_type": cur.get("plan_type"),  # Keep same plan type
                "plan_params": [],
                "category_type": None,
                "range_type": None,
                "base_value": None,
                "plan_base": None,
                "value_type": None,
                "valid_from": None,
                "valid_to": None,
                "absolute_calculation": False,
                "sequence": new_seq,
                "step": 0,
                "assignments": []
            }
            
            session["current_rule"] = new_condition
            session["stage"] = "rule_plan_params"  # Start fresh condition from plan_params
            session.modified = True
            
            # Show plan params for new condition
            params = opts.get("plan_params") or []
            return ask_options("rule_plan_params", params, "Choose Plan Params for the next condition:")
        
        else:  # sel.lower() == "no"
            # Save current rule and move to step/clone decision
            cur = session.get("current_rule")
            prules = pd.get("plan_rules", [])
            prules.append(dict(cur))
            pd["plan_rules"] = prules
            session["plan_data"] = pd
            
            session["stage"] = "rule_add_step_clone"
            session.modified = True
            return ask_options("rule_add_step_clone", ["Add Step", "Clone", "No"], "Would you like to Add Step or Clone this rule?")
    
    # FIXED: Add Step/Clone Logic
    if stage == "rule_add_step_clone":
        if not m:
            return ask_options("rule_add_step_clone", ["Add Step", "Clone", "No"], "Add Step / Clone / No")
        
        sel = resolve_input(m) or m
        
        if sel == "Add Step":
            # Get the last rule to base the step on
            prules = pd.get("plan_rules", [])
            if not prules:
                return {"response": "No rules available to add step to."}
            
            last_rule = prules[-1]
            # Create new step with step += 10 (backend only)
            new_step_rule = dict(last_rule)
            new_step_rule["step"] = last_rule.get("step", 0) + 10
            new_step_rule["assignments"] = []  # Clear assignments for new step
            
            session["current_rule"] = new_step_rule
            session["stage"] = "rule_absolute_calc"
            session.modified = True
            
            return ask_options("rule_absolute_calc", ON_OFF, "Step added. Absolute Calculation (On/Off):")
        
        elif sel == "Clone":
            # Get the last rule to clone
            prules = pd.get("plan_rules", [])
            if not prules:
                return {"response": "No rules available to clone."}
            
            last_rule = prules[-1]
            clone = dict(last_rule)
            clone["assignments"] = []  # Clear assignments for clone
            
            session["current_rule"] = clone
            session["stage"] = "rule_absolute_calc"
            session.modified = True
            
            return ask_options("rule_absolute_calc", ON_OFF, "Rule cloned. Absolute Calculation (On/Off):")
        
        elif sel == "No":
            # Move to assignments phase
            session["current_rule"] = None
            session["stage"] = "assignment_start"
            session.modified = True
            
            return {"response": "Plan Rules phase completed. Now starting Plan Rule Value Assignment phase.\nType 'start assignments' to begin assignments for rules."}
        
        else:
            return {"response": present_numbered_list(['Add Step','Clone','No'], "Add Step/Clone/No") + "\n\nInvalid selection."}
    
    # ---------- Phase 3: Assignments ----------
    if stage == "assignment_start":
        prules = pd.get("plan_rules", [])
        
        if not prules:
            return {"response": "No plan rules available to assign values to. You can add rules first."}
        
        # Handle numeric input for rule selection
        if m and m.isdigit():
            idx = int(m) - 1
            if 0 <= idx < len(prules):
                session["assignment_rule_index"] = idx
                session["stage"] = "assignment_add_yesno"
                session.modified = True
                return ask_options("assignment_add_yesno", YES_NO, f"Add assignment(s) for rule #{idx+1}?")
            return {"response": f"Invalid rule number. Choose 1-{len(prules)}."}
        
        # Handle "start assignments" or show rules list
        if not m or m.lower() == "start assignments":
            # show list of rules and ask to choose one - REMOVED sequence/step display
            lines = ["Choose rule to add assignments for (type number):"]
            for i, r in enumerate(prules, start=1):
                label = f"{i}. PlanType:{r.get('plan_type')} CategoryType:{r.get('category_type')} RangeType:{r.get('range_type')}"
                lines.append(label)
            session.modified = True
            return {"response": "\n".join(lines)}
        
        # Handle submit at assignment stage - UPDATED WITH BETTER ERROR HANDLING
        if m.lower() == "submit":
            payload = build_payload_from_session()
            try:
                resp = post_program_creation(payload)
                
                # Check if response contains error
                if isinstance(resp, dict) and resp.get("error"):
                    return {"response": f"Error submitting plan: {resp.get('error')}. Status: {resp.get('status_code', 'unknown')}"}
                
                # clear session on success
                session.clear()
                return {"response": f"Plan submitted successfully! Server response: {resp}"}
            except Exception as e:
                logger.error(f"[submit] error: {e}")
                return {"response": f"Error submitting plan: {e}. Check server logs and ensure the API server is running."}
        
        return {"response": "Type a rule number to add assignments or 'submit' to submit the plan."}
    
    if stage == "assignment_add_yesno":
        if not m:
            return ask_options("assignment_add_yesno", YES_NO, "Add assignment for this rule? (Yes/No)")
        
        sel = resolve_input(m) or m
        
        if sel not in YES_NO:
            return {"response": present_numbered_list(YES_NO, "Add assignment?") + "\n\nInvalid selection."}
        
        if sel.lower() == "no":
            # allow choose another rule or finish
            session["stage"] = "assignment_start"
            session.modified = True
            return {"response": "No assignment added. Choose another rule number to add assignment or type 'submit' to submit the plan."}
        
        # yes => start adding one assignment
        # initialize current_assignment
        session["current_assignment"] = {
            "combinations": [], 
            "tier_slabs": [], 
            "uom": None, 
            "currency": None, 
            "fromdate": pd.get("valid_from"), 
            "todate": pd.get("valid_to")
        }
        session["stage"] = "assignment_combinations" # first subtab
        session.modified = True
        
        # show params of the selected rule to choose combinations
        rule_idx = session.get("assignment_rule_index")
        prules = pd.get("plan_rules", [])
        rule = prules[rule_idx]
        params = rule.get("plan_params", [])
        
        if not params:
            # skip combinations if no plan params
            session["stage"] = "assignment_tierslab"
            session.modified = True
            return {"response": "No Plan Params for this rule. Proceeding to Tier Slab input."}
        
        # present params auto
        session["last_options"] = params
        session["last_option_key"] = "assignment_param"
        session.modified = True
        
        return {"response": present_numbered_list(params, "Choose a Plan Param to add combination for (type number or name), or type 'done' to skip combinations:")}
    
    if stage == "assignment_combinations":
        rule_idx = session.get("assignment_rule_index")
        prules = pd.get("plan_rules", [])
        
        if rule_idx is None or rule_idx >= len(prules):
            session["stage"] = "assignment_start"
            session.modified = True
            return {"response": "Invalid rule selection. Choose a rule first."}
        
        rule = prules[rule_idx]
        params = rule.get("plan_params", [])
        
        if not params:
            session["stage"] = "assignment_tierslab"
            session.modified = True
            return {"response": "No params. Proceeding to Tier Slab input."}
        
        if not m:
            # re-show params
            return {"response": present_numbered_list(params, "Choose a Plan Param to add combination for (or 'done' to finish combinations):")}
        
        if m.lower() == "done":
            session["stage"] = "assignment_tierslab"
            session.modified = True
            return {"response": "Moving to Tier Slab entry. Enter slab as 'from,to,commission'. Type 'none' to skip tier slab."}
        
        # user chose a param (single)
        chosen_param = resolve_choice(m, params) or m
        
        if chosen_param not in params:
            return {"response": present_numbered_list(params, "Choose a Plan Param:") + "\n\nInvalid param selection."}
        
        # fetch dropdown options for this param using assignee_objects
        ass_objs = opts.get("assignee_objects") or []
        combo_options = get_param_options_for_combination(chosen_param, ass_objs)
        
        if not combo_options:
            # ask for manual value entry
            session["pending_param"] = chosen_param
            session["stage"] = "assignment_combination_manual_value"
            session.modified = True
            return {"response": f"No dropdown options found for '{chosen_param}'. Type the value manually (exact string), or type 'skip' to skip this param."}
        
        # present combo options
        labels = [o.get("label") for o in combo_options]
        
        # store combo_options to session for resolving numeric -> label
        session["combo_options_for_param"] = combo_options
        session["pending_param"] = chosen_param
        session["last_options"] = labels
        session["last_option_key"] = "combo_values"
        session["stage"] = "assignment_combination_value"
        session.modified = True
        
        return {"response": present_numbered_list(labels, f"Options for {chosen_param}:") + "\n\nSelect option by number or exact name, or type 'skip' to skip this param."}
    
    if stage == "assignment_combination_manual_value":
        if not m:
            return {"response": "Type manual value for the pending param or 'skip'."}
        
        if m.lower() == "skip":
            session["stage"] = "assignment_combinations"
            session.modified = True
            return {"response": "Skipped. Choose another plan param or type 'done' to proceed."}
        
        # add manual combination
        pending = session.get("pending_param")
        ca = session.get("current_assignment", {})
        ca.setdefault("combinations", []).append({pending: m})
        session["current_assignment"] = ca
        
        # go back to param selection
        session["stage"] = "assignment_combinations"
        session.modified = True
        return {"response": f"Added {pending} -> {m}. Add another param or type 'done' to finish combinations."}
    
    if stage == "assignment_combination_value":
        if not m:
            labels = session.get("last_options") or []
            return {"response": present_numbered_list(labels, "Choose option:")}
        
        if m.lower() == "skip":
            session["stage"] = "assignment_combinations"
            session.modified = True
            return {"response": "Skipped param. Choose another param or type 'done'."}
        
        labels = session.get("last_options") or []
        resolved = resolve_choice(m, labels)
        
        if not resolved:
            return {"response": present_numbered_list(labels, "Choose option:") + "\n\nInvalid selection."}
        
        pending = session.get("pending_param")
        
        # store in current_assignment
        ca = session.get("current_assignment", {})
        ca.setdefault("combinations", []).append({pending: resolved})
        session["current_assignment"] = ca
        
        session["stage"] = "assignment_combinations"
        session.modified = True
        
        return {"response": f"Added combination {pending} -> {resolved}. Choose another plan param or type 'done' to finish combinations."}
    
    # FIXED: Tier Slab Logic - Only ONE slab per assignment
    if stage == "assignment_tierslab":
        if not m:
            return {"response": "Enter tier slab in format 'from,to,commission' (e.g., 0,1000,5). Type 'none' to skip tier slab."}
        
        if m.lower() == "none":
            # skip tier slabs
            session["stage"] = "assignment_uom"
            session.modified = True
            return ask_options("assignment_uom", [option_label(u) for u in (opts.get("uom") or [])], "Choose UOM (or type 'skip'):")
        
        # parse single slab
        try:
            fr, to, comm = [x.strip() for x in m.split(",")]
        except:
            return {"response": "Invalid format. Use: from,to,commission (e.g., 0,1000,5)."}
        
        ca = session.get("current_assignment", {})
        # Store only ONE tier slab (replace any existing ones)
        ca["tier_slabs"] = [{"fromValue": fr, "toValue": to, "commission": comm}]
        session["current_assignment"] = ca
        session["stage"] = "assignment_uom"  # Move directly to UOM
        session.modified = True
        
        return ask_options("assignment_uom", [option_label(u) for u in (opts.get("uom") or [])], "Tier slab saved. Choose UOM (or type 'skip'):")
    
    if stage == "assignment_uom":
        if not m:
            return ask_options("assignment_uom", [option_label(u) for u in (opts.get("uom") or [])], "Choose UOM (or type 'skip'):")
        
        if m.lower() == "skip":
            session["stage"] = "assignment_currency"
            session.modified = True
            return ask_options("assignment_currency", [option_label(c) for c in (opts.get("currency") or [])], "Choose Currency (or type 'skip'):")
        
        uoms_list = [option_label(u) for u in (opts.get("uom") or [])]
        sel = resolve_choice(m, uoms_list) or m
        
        if sel not in uoms_list:
            return {"response": present_numbered_list(uoms_list, "Choose UOM:") + "\n\nInvalid selection."}
        
        ca = session.get("current_assignment")
        ca["uom"] = sel
        session["current_assignment"] = ca
        session["stage"] = "assignment_currency"
        session.modified = True
        
        return ask_options("assignment_currency", [option_label(c) for c in (opts.get("currency") or [])], "Choose Currency (or type 'skip'):")
    
    if stage == "assignment_currency":
        if not m:
            return ask_options("assignment_currency", [option_label(c) for c in (opts.get("currency") or [])], "Choose Currency (or type 'skip'):")
        
        if m.lower() == "skip":
            # finalize assignment with defaults
            idx = session.get("assignment_rule_index")
            prules = pd.get("plan_rules", [])
            
            if idx is None or idx >= len(prules):
                return {"response": "Invalid rule index."}
            
            ca = session.get("current_assignment", {})
            prules[idx].setdefault("assignments", []).append(ca)
            pd["plan_rules"] = prules
            session["plan_data"] = pd
            session["current_assignment"] = None
            session["stage"] = "assignment_add_another"
            session.modified = True
            
            return ask_options("assignment_add_another", YES_NO, "Assignment saved. Add another assignment to this rule?")
        
        curs_list = [option_label(c) for c in (opts.get("currency") or [])]
        sel = resolve_choice(m, curs_list) or m
        
        if sel not in curs_list:
            return {"response": present_numbered_list(curs_list, "Choose Currency:") + "\n\nInvalid selection."}
        
        ca = session.get("current_assignment", {})
        ca["currency"] = sel
        session["current_assignment"] = ca
        
        # finalize assignment -> append to selected rule
        idx = session.get("assignment_rule_index")
        prules = pd.get("plan_rules", [])
        prules[idx].setdefault("assignments", []).append(ca)
        pd["plan_rules"] = prules
        session["plan_data"] = pd
        session["current_assignment"] = None
        session["stage"] = "assignment_add_another"
        session.modified = True
        
        return ask_options("assignment_add_another", YES_NO, "Assignment saved. Add another assignment to this rule?")
    
    if stage == "assignment_add_another":
        if not m:
            return ask_options("assignment_add_another", YES_NO, "Add another assignment?")
        
        sel = resolve_input(m) or m
        
        if sel not in YES_NO:
            return {"response": present_numbered_list(YES_NO, "Add another assignment?") + "\n\nInvalid selection."}
        
        if sel.lower() == "yes":
            # restart assignment for same rule
            session["current_assignment"] = {
                "combinations": [], 
                "tier_slabs": [], 
                "uom": None, 
                "currency": None, 
                "fromdate": pd.get("valid_from"), 
                "todate": pd.get("valid_to")
            }
            session["stage"] = "assignment_combinations"
            session.modified = True
            
            # show params again
            rule_idx = session.get("assignment_rule_index")
            rule = pd.get("plan_rules")[rule_idx]
            params = rule.get("plan_params", [])
            
            if not params:
                session["stage"] = "assignment_tierslab"
                session.modified = True
                return {"response": "No Plan Params for this rule. Proceeding to Tier Slab input."}
            
            session["last_options"] = params
            session["last_option_key"] = "assignment_param"
            session.modified = True
            
            return {"response": present_numbered_list(params, "Choose a Plan Param to add combination for (or 'done'):")}
        
        else:
            # ask user choose another rule or finish and submit
            session["stage"] = "assignment_start"
            session.modified = True
            return {"response": "Assignment(s) completed for this rule. You can choose another rule number to add assignments, or type 'submit' to submit the plan."}
    
    # ---------- Final submit ----------
    if m.lower() == "submit":
        # prepare payload and post
        payload = build_payload_from_session()
        try:
            resp = post_program_creation(payload)
            
            # Check if response contains error
            if isinstance(resp, dict) and resp.get("error"):
                return {"response": f"Error submitting plan: {resp.get('error')}. Status: {resp.get('status_code', 'unknown')}"}
            
            # clear session on success
            session.clear()
            return {"response": f"Plan submitted successfully! Server response: {resp}"}
        except Exception as e:
            logger.error(f"[submit] error: {e}")
            return {"response": f"Error submitting plan: {e}. Check server logs and ensure the API server is running."}
    
    # fallback while in flow
    return {"response": "In plan creation flow. Type 'help' or follow the prompts. Type 'cancel' to abort."}

# --- COMPLETELY REWRITTEN: Build payload for POST from session data ---
def build_payload_from_session():
    pd = session.get("plan_data", {})
    opts = session.get("options", {})
    
    calc_name = pd.get("calculation_schedule")
    pay_name = pd.get("payment_schedule")
    assignee_name = pd.get("assignee_name")
    
    # convert schedule names to ids if available
    calc_id = None; pay_id = None
    calc_label = ""; pay_label = ""
    for s in (opts.get("schedules") or []):
        if option_label(s).lower() == (calc_name or "").lower():
            calc_id = s.get("id")
            calc_label = option_label(s)
        if option_label(s).lower() == (pay_name or "").lower():
            pay_id = s.get("id")
            pay_label = option_label(s)
    
    # Find assignee details from assignee_objects
    assignee_id = None
    table_id = None
    table_name = ""
    for a in (opts.get("assignee_objects") or []):
        assignee_label = a.get("objectType") or a.get("label") or a.get("objectName") or a.get("name")
        if assignee_label and assignee_label.lower() == (assignee_name or "").lower():
            assignee_id = a.get("id")
            table_id = a.get("id")  # Using same ID for both
            table_name = a.get("tableName") or "hierarchy_master_instance"
            break
    
    # resolve plan types id
    def plan_type_id_by_name(name):
        for p in fetch_plan_types():
            if (p.get("planName") or "").strip().lower() == (name or "").strip().lower():
                return p.get("id")
        return None
    
    # Convert date format from YYYY-MM-DD to YYYY-MM-DDTHH:MM:SS
    def format_datetime(date_str, is_end_date=False):
        if not date_str:
            return ""
        try:
            # Parse the input date
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if is_end_date:
                return dt.strftime("%Y-%m-%dT23:59:59")
            else:
                return dt.strftime("%Y-%m-%dT00:00:00")
        except:
            return date_str
    
    plan_conditions = []
    
    for r in pd.get("plan_rules", []):
        plan_type_name = r.get("plan_type")
        plan_type_id = plan_type_id_by_name(plan_type_name)
        
        # Create JsonData structure
        json_data_list = []
        
        for a in r.get("assignments", []):
            # Build tiers structure
            tiers = []
            for ts in a.get("tier_slabs", []):
                tier = {
                    "from_value": ts.get("fromValue", ""),
                    "to_value": ts.get("toValue", "") if ts.get("toValue") else "",
                    "commission": ts.get("commission", "")
                }
                tiers.append(tier)
            
            # Build PlanValueRangesDto structure
            plan_value_ranges = {}
            
            # Initialize all parameter arrays
            plan_value_ranges["BusinessPartner"] = []
            plan_value_ranges["Product"] = []
            plan_value_ranges["Territory"] = []
            plan_value_ranges["Plant"] = []
            plan_value_ranges["Group"] = []
            
            # Map combinations to appropriate parameter arrays
            ass_objs = opts.get("assignee_objects") or []
            for combo in a.get("combinations", []):
                for param, label in combo.items():
                    # Find the ID for this label
                    resolved_id = None
                    for ao in ass_objs:
                        lbl = ao.get("objectType") or ao.get("label") or ao.get("name")
                        if lbl and lbl.strip().lower() == str(label).strip().lower() and (ao.get("type") or "").strip().lower() == param.strip().lower():
                            resolved_id = ao.get("id")
                            break
                    
                    if resolved_id:
                        # Map parameter to correct field name
                        param_lower = param.lower()
                        if param_lower == "businesspartner":
                            plan_value_ranges["BusinessPartner"].append(resolved_id)
                        elif param_lower == "product":
                            plan_value_ranges["Product"].append(resolved_id)
                        elif param_lower == "territory":
                            plan_value_ranges["Territory"].append(resolved_id)
                        elif param_lower == "plant":
                            plan_value_ranges["Plant"].append(resolved_id)
                        elif param_lower == "group":
                            plan_value_ranges["Group"].append(resolved_id)
            
            # Get currency ID
            cur_val = a.get("currency")
            cur_id = 78  # Default fallback
            for c in (opts.get("currency") or []):
                if (c.get("currencyCode") or c.get("currencyName") or "").strip().lower() == (cur_val or "").strip().lower() or option_label(c).lower() == (cur_val or "").lower():
                    cur_id = c.get("id")
                    break
            
            json_data_item = {
                "commission": "",
                "tiers": tiers,
                "PlanValueRangesDto": [plan_value_ranges],
                "currency": cur_id
            }
            json_data_list.append(json_data_item)
        
        # Build PlanAssignments structure (for backward compatibility)
        assignments_out = []
        for a in r.get("assignments", []):
            # Build plan_ranges with correct field names
            plan_ranges = []
            for ts in a.get("tier_slabs", []):
                plan_ranges.append({
                    "from_value": int(ts.get("fromValue", 0)) if ts.get("fromValue", "").isdigit() else ts.get("fromValue", ""),
                    "to_value": int(ts.get("toValue", 0)) if ts.get("toValue", "") and ts.get("toValue", "").isdigit() else (ts.get("toValue") if ts.get("toValue") else None),
                    "commission": int(ts.get("commission", 0)) if ts.get("commission", "").isdigit() else ts.get("commission", "")
                })
            
            # Get currency ID
            cur_val = a.get("currency")
            cur_id = 78  # Default fallback
            for c in (opts.get("currency") or []):
                if (c.get("currencyCode") or c.get("currencyName") or "").strip().lower() == (cur_val or "").strip().lower() or option_label(c).lower() == (cur_val or "").lower():
                    cur_id = c.get("id")
                    break
            
            assignments_out.append({
                "plan_assignments": [],  # Empty as per example
                "plan_ranges": plan_ranges,
                "currency": cur_id,
                "fromdate": r.get("valid_from", ""),
                "todate": r.get("valid_to", "")
            })
        
        rule_obj = {
            "PlanType": plan_type_name,
            "PlanTypeMasterId": plan_type_id,
            "Sequence": r.get("sequence", 100),
            "Step": r.get("step", 10),
            "PlanDescription": f"{plan_type_name} plan",  # Generated description
            "PlanParamsJson": json.dumps([p.lower() for p in r.get("plan_params", [])]),  # Lowercase param names
            "Tiered": r.get("category_type") == "Tiered",
            "CategoryType": r.get("category_type"),
            "RangeType": r.get("range_type"),
            "ValueType": r.get("value_type"),
            "ValidFrom": r.get("valid_from", ""),
            "ValidTo": r.get("valid_to", ""),
            "PlanBase": r.get("plan_base"),
            "BaseValue": r.get("base_value"),
            "JsonData": json_data_list,
            "ShowAssignment": True,
            "isStep": r.get("step", 0) > 0,
            "AbsoluteCalculation": r.get("absolute_calculation", False),
            "PlanAssignments": assignments_out
        }
        plan_conditions.append(rule_obj)
    
    # Build final payload with correct structure
    payload = {
        "id": 0,
        "OrgCode": "",
        "PaymentSchedule": pay_id,
        "ProgramName": pd.get("plan_name"),
        "CalculationSchedule": calc_id,
        "ValidFrom": format_datetime(pd.get("valid_from"), False),
        "ValidTo": format_datetime(pd.get("valid_to"), True),
        "ReviewedBy": "",
        "ReviewStatus": "",
        "ObjectType": pd.get("object_type"),
        "TableId": table_id,
        "AssigneeId": assignee_id,
        "TableName": table_name,
        "AssigneeName": assignee_name,
        "PlanConditionsDto": plan_conditions,
        "CalculationScheduleLabel": calc_label,
        "PaymentScheduleLabel": pay_label,
        "Id": 0
    }
    
    client_id = session.get("client_id") or getattr(request, "client_id", None)
    org_id = session.get("org_id") or None
    client_code = session.get("client_code") or None
    org_code = session.get("org_code") or None

    if client_id:
        payload["client_id"] = client_id    # If used in your API/DB
    if org_id:
        payload["org_id"] = org_id          # If used in your API/DB
    if client_code:
        payload["client_code"] = client_code
    if org_code:
        payload["org_code"] = org_code
        
    logger.info("[build_payload_from_session] NEW payload preview: " + json.dumps(payload)[:1500])
    return payload

# --- Chat endpoint - UPDATED WITH INTENT RECOGNITION AND RAG INTEGRATION ---
@app.route("/chat", methods=["POST"])
@verify_jwt_token
def chat_endpoint():
    data = request.json or {}
    user_msg = (data.get("message") or "").strip()
    user_id = data.get("user_id") or data.get("userId") or None
    org_id = getattr(request, "org_id", None)
    client_id = get_client_id_for_org(org_id) if org_id else None
    
    #logger.info(f"[chat] User message: '{user_msg}' user_id: {user_id}")
    logger.info(f"[chat] User message: '{user_msg}' user_id: {user_id},client_id: {client_id}") #client_id: {getattr(request,'client_id',None)}")

    # If user_id provided, load server-side saved session state (so refresh/browser-independent)
    if user_id:
        state = load_user_state(user_id) or {}
        # copy keys into flask session
        session.clear()
        for k, v in state.items():
            session[k] = v
        session.modified = True
    
    # Commands: cancel/clear session
    if user_msg.lower() in ["cancel", "clear session", "/clear_session"]:
        session.clear()
        if user_id:
            clear_user_state(user_id)
        return jsonify({"response": "Session cleared. You can now ask questions or start plan creation."})
    
    # Check if user is already in plan creation mode
    if session.get("mode") == "plan_creation":
        logger.info("🎯 [PLAN] User is in plan creation mode, continuing flow")
        resp = handle_message_in_flow(user_msg)
        # save server-side if user_id provided
        if user_id:
            save_user_state(user_id, dict(session))
        return jsonify(resp)
    
    # Intent recognition for new conversations
    if is_plan_creation_intent(user_msg):
        logger.info("🎯 [PLAN] Starting plan creation flow")
        start_plan_creation_session()
        # persist if user_id provided
        if user_id:
            save_user_state(user_id, dict(session))
        return jsonify({"response": "Started plan creation. Please enter Plan Name:"})
    
    if is_database_query_intent(user_msg):
        logger.info("🗄️ [DB] Processing database query")
        try:
            db_response = handle_database_query(user_msg)
            
            # persist session state if requested
            if user_id:
                save_user_state(user_id, dict(session))
            
            return jsonify({"response": db_response})
            
        except Exception as e:
            logger.error(f"❌ [DB] Error in database processing: {e}")

    # Fall back to RAG for general queries
    logger.info("🧠 [RAG] Processing general query")
    try:
        rag_response = handle_rag_query(user_msg)
        
        # persist session state if requested
        if user_id:
            save_user_state(user_id, dict(session))
        
        return jsonify({"response": rag_response})
        
    except Exception as e:
        logger.error(f"❌ [RAG] Error in RAG processing: {e}")
        
        # persist session state if requested
        if user_id:
            save_user_state(user_id, dict(session))
        
        return jsonify({"response": f"I encountered an error while processing your question: {str(e)}. Please try again or contact support."})

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
        "https://......./api/authenticate",
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
    return jsonify({'success': True})

# --- Run server ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"🚀 Flask running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
