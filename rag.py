import os
import re
import json
import string
import PyPDF2
import pandas as pd
import pytesseract
from PIL import Image
import warnings
from docx import Document as DocxDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple


warnings.filterwarnings("ignore", category=UserWarning)


_vector_store_cache = None
_source_chunks = []


# 🔁 Used to reset on upload
def clear_vector_cache():
    global _vector_store_cache, _source_chunks
    print("🔄 Clearing vector cache...")
    _vector_store_cache = None
    _source_chunks = []
    print("✅ Vector cache cleared successfully")


# 📱 WhatsApp chat parsing
def preprocess_whatsapp_chat(text: str) -> list[str]:
    pattern = r"^\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}"
    messages = []
    current = ""

    for line in text.splitlines():
        if re.match(pattern, line):
            if current:
                messages.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        messages.append(current.strip())

    clean_msgs = []
    for msg in messages:
        if "<Media omitted>" in msg or re.fullmatch(r"[👆🏼⬆️]+", msg.strip()):
            continue
        clean_msgs.append(msg)
    return clean_msgs


# 📂 Load text content from various files
def load_file_texts(folder_path: str) -> List[Tuple[str, str]]:
    print(f"📂 Loading files from: {folder_path}")
    texts = []
    files_processed = 0
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder does not exist: {folder_path}")
        return texts
        
    for fn in os.listdir(folder_path):
        path = os.path.join(folder_path, fn)
        if not os.path.isfile(path):
            continue
            
        low = fn.lower()
        files_processed += 1
        print(f"📄 Processing file: {fn}")

        try:
            if low.endswith(".pdf"):
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    content = "\n".join(pages)
                    texts.append((fn, content))
                    print(f"   ✅ PDF processed: {len(content)} chars")

            elif low.endswith(".docx"):
                doc = DocxDocument(path)
                paras = [para.text for para in doc.paragraphs if para.text.strip()]
                content = "\n".join(paras)
                texts.append((fn, content))
                print(f"   ✅ DOCX processed: {len(content)} chars")

            elif low.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                if " - " in raw and re.match(r"^\d{1,2}/\d{1,2}/\d{2}", raw.strip()):
                    msgs = preprocess_whatsapp_chat(raw)
                    for m in msgs:
                        texts.append((fn, m))
                    print(f"   ✅ WhatsApp chat processed: {len(msgs)} messages")
                else:
                    texts.append((fn, raw))
                    print(f"   ✅ TXT processed: {len(raw)} chars")

            elif low.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    texts.append((fn, json.dumps(data)))
                elif isinstance(data, list):
                    for item in data:
                        texts.append((fn, json.dumps(item)))
                print(f"   ✅ JSON processed")

            elif low.endswith((".xls", ".xlsx")):
                df_dict = pd.read_excel(path, sheet_name=None)
                for sheet, df in df_dict.items():
                    texts.append((f"{fn} [{sheet}]", df.to_csv(index=False)))
                print(f"   ✅ Excel processed: {len(df_dict)} sheets")

            elif low.endswith((".png", ".jpg", ".jpeg", ".tiff")):
                ocr = pytesseract.image_to_string(Image.open(path))
                texts.append((fn, ocr))
                print(f"   ✅ Image OCR processed: {len(ocr)} chars")

        except Exception as e:
            print(f"❌ Error loading {fn}: {e}")
    
    print(f"📊 Total files processed: {files_processed}, Text chunks created: {len(texts)}")
    return texts


# 🧠 Embed & store in FAISS
def get_vector_store(texts: List[Tuple[str, str]]) -> FAISS:
    global _vector_store_cache, _source_chunks
    
    print("🧠 Building vector store...")
    
    if not texts:
        print("❌ No texts provided to build vector store")
        _vector_store_cache = None
        _source_chunks = []
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks, metadatas = [], []

    for filename, content in texts:
        if content and content.strip():  # Only process non-empty content
            split_chunks = splitter.split_text(content)
            for chunk in split_chunks:
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append(chunk)
                    metadatas.append({"source": filename})

    if not chunks:
        print("❌ No valid chunks created from texts")
        _vector_store_cache = None
        _source_chunks = []
        return None

    print(f"📝 Created {len(chunks)} chunks from {len(texts)} documents")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vector_store_cache = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
        _source_chunks = list(zip(chunks, metadatas))
        print(f"✅ Vector store built successfully with {len(chunks)} embeddings")
        return _vector_store_cache
    except Exception as e:
        print(f"❌ Error building vector store: {e}")
        _vector_store_cache = None
        _source_chunks = []
        return None


def normalize_text(text: str) -> str:
    """Normalize text by replacing curly quotes with straight ones and reducing multiple spaces."""
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 🧹 IMPROVED: Keyword relevance filter with better matching
def is_relevant(text: str, query: str) -> bool:
    """
    Check if the text is relevant to the query with improved keyword matching.
    """
    text = normalize_text(text)
    query = normalize_text(query)
    
    qwords = [w for w in query.lower().split() if len(w) > 2]
    text_words = [w.strip(string.punctuation).lower() for w in text.split()]
    
    # Create word variations for better matching
    word_variations = {
        'favourite': ['fav', 'favorite', 'favourite'],
        'color': ['colour', 'color'],
        'ash': ['ash'],
        'plan': ['plan', 'plans', 'planning'],
        'commission': ['commission', 'commissions'],
        'create': ['create', 'creation', 'creating', 'created'],
        'earned': ['earned', 'earn', 'earning'],
        'year': ['year', 'years'],
        # Add more variations as needed
    }
    
    # Check for exact matches first
    exact_matches = 0
    for word in qwords:
        if word in text_words:
            exact_matches += 1
    
    # Check for word variations
    variation_matches = 0
    for qword in qwords:
        if qword in text_words:
            continue  # Already counted in exact matches
        
        # Check if any variation of the query word exists in text
        for key, variations in word_variations.items():
            if qword in variations:
                if any(var in text_words for var in variations):
                    variation_matches += 1
                    break
        
        # Check for partial matches (substring matching for longer words)
        if len(qword) > 4:
            for text_word in text_words:
                if len(text_word) > 4 and (qword in text_word or text_word in qword):
                    variation_matches += 0.5  # Partial credit
                    break
    
    total_matches = exact_matches + variation_matches
    
    # More lenient matching: require at least 30% of query words to match
    relevance_threshold = max(1, len(qwords) * 0.3)
    
    is_match = total_matches >= relevance_threshold
    
    # Debug logging
    print(f"   🔍 Relevance check: '{query}' in text snippet")
    print(f"      Query words: {qwords}")
    print(f"      Exact matches: {exact_matches}, Variation matches: {variation_matches}")
    print(f"      Total matches: {total_matches}, Threshold: {relevance_threshold}")
    print(f"      Result: {'✅ RELEVANT' if is_match else '❌ NOT RELEVANT'}")
    
    return is_match


def extract_clean_answer_from_chunks(chunks: List[Tuple[str, str]], query: str) -> str:
    """
    Extract a clean answer from the chunks without source file references
    """
    if not chunks:
        return "I couldn't find relevant information to answer your question."
    
    # Extract just the content without source references
    clean_content_pieces = []
    for chunk_text, source in chunks:
        if chunk_text.strip():
            clean_content_pieces.append(chunk_text.strip())
    
    if not clean_content_pieces:
        return "I found some information but couldn't extract a clear answer."
    
    # For simple queries, try to find the most direct answer
    query_lower = query.lower()
    
    # Look for direct answers in the content
    best_answer = None
    for piece in clean_content_pieces:
        piece_lower = piece.lower()
        
        # For "what is" questions, look for direct statements
        if "what is" in query_lower or "what are" in query_lower:
            # Look for sentences that might contain the answer
            sentences = piece.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_lower.split() if len(word) > 2):
                    if not best_answer or len(sentence.strip()) < len(best_answer):
                        best_answer = sentence.strip()
        
        # For other questions, take the most relevant piece
        elif any(word in piece_lower for word in query_lower.split() if len(word) > 2):
            if not best_answer or len(piece) < len(best_answer):
                best_answer = piece
    
    if best_answer:
        # Clean up the answer
        best_answer = best_answer.strip()
        if not best_answer.endswith('.'):
            best_answer += '.'
        return best_answer
    
    # Fallback: return first meaningful piece of content
    first_piece = clean_content_pieces[0][:300]  # Limit length
    if len(clean_content_pieces[0]) > 300:
        first_piece += "..."
    
    return first_piece


# 🧠 UPDATED: Retrieve matched context - now returns clean answer instead of formatted context
def retrieve_context(query: str, folder_path: str, k: int = 7) -> Tuple[str, List[Tuple[str, str]]]:
    global _vector_store_cache, _source_chunks

    print(f"🔍 Retrieving context for query: '{query}'")
    
    # If vector store is not cached, build it
    if _vector_store_cache is None:
        print("🔄 Vector store not found, building from documents...")
        texts = load_file_texts(folder_path)
        if not texts:
            print("❌ No texts found to build vector store")
            return "", []
        
        # Build and cache the vector store
        vs = get_vector_store(texts)
        if vs is None:
            print("❌ Failed to build vector store")
            return "", []
        
        # The vector store is now cached in the global variable
        print("✅ Vector store built and cached")
    
    # Use the cached vector store
    vs = _vector_store_cache
    if vs is None:
        print("❌ Vector store is None, cannot retrieve context")
        return "", []

    try:
        docs_and_scores = vs.similarity_search_with_score(query, k=k)
        print(f"📊 Found {len(docs_and_scores)} potential matches")
    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        return "", []

    relevant_chunks = []
    RELEVANCE_THRESHOLD = 2.5  # More lenient threshold

    for doc, score in docs_and_scores:
        source = doc.metadata.get("source", "Unknown")
        print(f"   📄 {source}: score={score:.3f}")
        
        # IMPROVED LOGIC: For very good scores (< 0.5), skip keyword check
        if score < 0.5:  # Excellent match - trust the embedding similarity
            relevant_chunks.append((doc.page_content, source))
            print(f"   ✅ Added to context (excellent score: {score:.3f})")
        elif score < RELEVANCE_THRESHOLD:
            # For decent scores, check keyword relevance
            if is_relevant(doc.page_content, query):
                relevant_chunks.append((doc.page_content, source))
                print(f"   ✅ Added to context (good score + relevant: {score:.3f})")
            else:
                print(f"   ❌ Filtered out (score {score:.3f} but not keyword relevant)")
        else:
            print(f"   ❌ Filtered out (score {score:.3f} >= threshold {RELEVANCE_THRESHOLD})")

    print(f"📋 Final context chunks: {len(relevant_chunks)}")
    
    # If still no results, be more lenient
    if not relevant_chunks:
        print("❌ No relevant chunks found with strict criteria")
        print("🔄 Trying with very lenient matching...")
        for doc, score in docs_and_scores[:3]:  # Try top 3 results
            source = doc.metadata.get("source", "Unknown")
            if score < 3.0:  # Very lenient threshold
                relevant_chunks.append((doc.page_content, source))
                print(f"   ✅ Added with very lenient matching: {source} (score: {score:.3f})")
    
    if not relevant_chunks:
        print("❌ Still no relevant chunks found")
        return "", []

    # Sort by score (lower is better for FAISS)
    if len(relevant_chunks) > 1:
        # Get scores for sorting
        chunk_scores = []
        for chunk_text, source in relevant_chunks:
            for doc, score in docs_and_scores:
                if doc.page_content == chunk_text and doc.metadata.get("source") == source:
                    chunk_scores.append((chunk_text, source, score))
                    break
        
        # Sort by score and take best chunks
        chunk_scores.sort(key=lambda x: x[2])  # Sort by score (ascending - lower is better)
        relevant_chunks = [(text, source) for text, source, score in chunk_scores[:5]]  # Take top 5

    # LOG DETAILED CONTEXT AND SOURCES (for debugging)
    print("🔍 DETAILED CONTEXT WITH SOURCES:")
    for i, (chunk_text, source) in enumerate(relevant_chunks, 1):
        print(f"   📄 Source {i}: {source}")
        print(f"   📝 Content: {chunk_text[:200]}{'...' if len(chunk_text) > 200 else ''}")

    # LOG FULL FORMATTED CONTEXT (for debugging)
    formatted_context = [f"[{source}]\n{chunk.strip()}" for chunk, source in relevant_chunks]
    full_formatted_context = "\n\n---\n\n".join(formatted_context)
    print(f"📋 FULL FORMATTED CONTEXT:\n{full_formatted_context}")

    # EXTRACT CLEAN ANSWER (what gets returned to frontend)
    clean_answer = extract_clean_answer_from_chunks(relevant_chunks, query)
    
    print(f"✅ Clean answer extracted: {len(clean_answer)} characters")
    print(f"💬 Frontend response: {clean_answer}")
    
    return clean_answer, relevant_chunks
