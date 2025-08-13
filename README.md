The browser agent is momentarily unavailable, but I won’t block you. I’ll provide a finalized README you can paste directly into the editor you already have open. It includes Windows/macOS/Linux setup, Tesseract install commands, a sample plan-creation payload, and a screenshots section.

Copy everything below into README.md and commit with message:
docs: add comprehensive README with setup, payload example, and screenshots section

Commission Dashboard Assistant (RAG)

A production-ready assistant for creating commission plans and answering domain questions using Retrieval-Augmented Generation (RAG). It provides a stateful, guided plan-creation flow backed by your APIs, plus a document QA pipeline that indexes local files (PDF, DOCX, Excel/CSV, TXT including WhatsApp format, JSON, and images via OCR) using FAISS and Hugging Face embeddings.

Features

Guided, multi-phase plan creation flow with session persistence

RAG-powered Q&A over your local data/ folder

Robust document ingestion (PDF/DOCX/Excel/CSV/JSON/TXT/Images via OCR)

Vector store caching and one-click reset

Connectivity diagnostics endpoint and detailed logging

Clean HTML frontend with template rendering

Tech stack

Python 3.10+

Flask, Flask-Session

FAISS, Hugging Face sentence-transformers (all-MiniLM-L6-v2)

LangChain components

PyPDF2, python-docx, pandas, pytesseract, Pillow

requests, python-dotenv

Architecture

[Browser UI] -> Flask (app.py) -> Your Plan APIs
-> RAG (rag.py)
-> FAISS + Embeddings

Repository structure

app.py

rag.py

requirements.txt

.env

templates/

index.html

data/

README.md

.gitkeep

model/

README.md

.gitkeep

.gitignore

README.md

Core components

Flask application (app.py)

Session flow

Detects plan-creation intent and launches a step-by-step wizard

Continues the wizard if already in progress

Falls back to RAG Q&A for non-plan messages

Endpoints

GET / Serve templates/index.html

POST /chat Plan flow and RAG router

GET /_test_api Tests connectivity to your backend APIs

POST /_clear_cache Clears RAG vector cache

Backend API integrations

fetch_schedules, fetch_assignee_objects, fetch_plan_types, fetch_plan_params, fetch_uom, fetch_currency

post_program_creation(payload) with robust error handling

Payload builder

build_payload_from_session transforms session state into the exact server payload, including PlanConditionsDto and JsonData structures

RAG pipeline (rag.py)

Ingestion

PDF (PyPDF2)

DOCX (python-docx)

TXT (plain text; WhatsApp format parser)

JSON (dict/list)

Excel (xls/xlsx via pandas)

Images (png/jpg/jpeg/tiff via pytesseract OCR)

Chunking and embeddings

RecursiveCharacterTextSplitter (800/100 overlap)

sentence-transformers/all-MiniLM-L6-v2 + FAISS

Relevance

Combines vector similarity with keyword/variation checks

Lenient fallback if strict filtering yields no results

Output

retrieve_context(query, folder) -> (clean_answer, relevant_chunks)

clear_vector_cache() to force re-indexing

Frontend (templates/index.html)

Chat interface for both plan wizard and RAG Q&A

Modern, responsive markup

Environment variables (.env)

FLASK_SECRET_KEY Flask session secret

BACKEND_API_BASE_URL Your API base (default: https://localhost:8081)

JWT_TOKEN Bearer token for your APIs (optional but recommended)

DOCUMENTS_FOLDER Folder for RAG docs (use data)

Security and data hygiene

Do not commit secrets; .env is local only

Do not commit large or private data or model files; keep them out of Git

model/ stores local LLM artifacts (e.g., GGUF) for runtime mount/download

Quick start

Prerequisites

Python 3.10+ recommended

Tesseract OCR (required for image OCR)

Create and activate a virtual environment

macOS/Linux:

python3 -m venv .venv

source .venv/bin/activate

Windows:

python -m venv .venv

.venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

Configure environment

Create .env with:

FLASK_SECRET_KEY=your-strong-secret

BACKEND_API_BASE_URL=https://localhost:8081

JWT_TOKEN=your-jwt-token-if-needed

DOCUMENTS_FOLDER=data

Place documents in data/

Supported: PDF, DOCX, TXT (incl. WhatsApp), JSON, Excel/CSV, and images

Avoid pushing sensitive/large data into Git

Run the app

python app.py

Access at http://0.0.0.0:5000

Installing Tesseract OCR

Windows

winget:

winget install -e --id UB-Mannheim.TesseractOCR

Chocolatey:

choco install tesseract

After install, ensure Tesseract is in PATH or set TESSDATA_PREFIX if needed.

macOS

brew install tesseract

Ubuntu/Debian

sudo apt-get update

sudo apt-get install -y tesseract-ocr

Fedora

sudo dnf install -y tesseract

Endpoints

POST /chat

Body:
{
"message": "create plan",
"user_id": "optional-stable-user-id"
}

Behavior:

Starts or continues plan wizard; for non-plan queries, runs RAG Q&A

Response:
{
"response": "assistant reply"
}

GET /_test_api

Returns:
{
"api_base_url": "...",
"test_endpoint_status": 200,
"jwt_present": true,
"test_response": "..."
}

POST /_clear_cache

Returns:
{
"message": "Vector cache cleared successfully"
}

Plan creation flow (what to expect)

Phase 1: Plan metadata

Plan Name

Calculation Schedule

Payment Schedule

Assignee Name

Object Type (Invoices/Contracts/Sales Orders)

Valid From/To (YYYY-MM-DD)

Phase 2: Rules

Plan Type, Plan Params

Category Type (Flat/Slab/Tiered)

Range Type (Amount/Quantity)

Base Value (Gross/Net)

Plan Base (Revenue/Margin)

Value Type (Percentage/Amount)

Rule validity dates

Absolute Calculation On/Off

Optionally Add Condition, Add Step, or Clone

Phase 3: Assignments

Combinations for chosen params (resolved against assignee objects; manual allowed)

One tier slab (from,to,commission)

UOM and Currency

Save assignment(s), then submit

Sample payload (abbreviated)
{
"id": 0,
"OrgCode": "",
"PaymentSchedule": 12,
"ProgramName": "Q3 Incentives",
"CalculationSchedule": 34,
"ValidFrom": "2025-07-01T00:00:00",
"ValidTo": "2025-09-30T23:59:59",
"ReviewedBy": "",
"ReviewStatus": "",
"ObjectType": "Invoices",
"TableId": 99,
"AssigneeId": 456,
"TableName": "hierarchy_master_instance",
"AssigneeName": "North Region",
"PlanConditionsDto": [
{
"PlanType": "Revenue Target",
"PlanTypeMasterId": 7,
"Sequence": 100,
"Step": 0,
"PlanDescription": "Revenue Target plan",
"PlanParamsJson": "["territory","product"]",
"Tiered": true,
"CategoryType": "Tiered",
"RangeType": "Amount",
"ValueType": "Percentage",
"ValidFrom": "2025-07-01",
"ValidTo": "2025-09-30",
"PlanBase": "Revenue",
"BaseValue": "Gross",
"JsonData": [
{
"commission": "",
"tiers": [
{ "from_value": "0", "to_value": "100000", "commission": "5" }
],
"PlanValueRangesDto": [
{
"BusinessPartner": [],
"Product": [311,
"Territory": ,
"Plant": [],
"Group": []
}
],
"currency": 78
}
],
"ShowAssignment": true,
"isStep": false,
"AbsoluteCalculation": false,
"PlanAssignments": [
{
"plan_assignments": [],
"plan_ranges": [
{ "from_value": 0, "to_value": 100000, "commission": 5 }
],
"currency": 78,
"fromdate": "2025-07-01",
"todate": "2025-09-30"
}
]
}
],
"CalculationScheduleLabel": "Monthly",
"PaymentScheduleLabel": "Monthly",
"Id": 0
}

RAG usage

Add files into data/

POST /chat with a domain question, e.g.:

{ "message": "Summarize the Q2 commission tiers for North Region" }

First call builds the index; subsequent calls reuse cache

If results seem stale, call POST /_clear_cache and retry

Troubleshooting

API 401/404/500: verify BACKEND_API_BASE_URL, JWT_TOKEN, and endpoint correctness

No RAG results: ensure files exist in data/ with supported formats; try clearing cache

OCR issues: verify Tesseract installation and PATH; try higher-quality images

FAISS install problems: use pinned faiss-cpu in requirements; consult platform docs

Performance tips

Keep chunks around 800 with 100 overlap; adjust per corpus

For large corpora, consider a persistent vector DB (Milvus/Chroma) and batch ingestion

Consider a stronger embedding model if recall is critical

Roadmap

Authn/Authz and user roles

Streaming responses and progress UI

Dockerfile and CI/CD

Pluggable vector stores and embedding models

Rich UI for rule/assignment editing

Background ingestion service
