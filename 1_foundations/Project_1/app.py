from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import re
import hashlib
import time
import random
from pathlib import Path

# === [ENV LOADING] ===
# Load environment variables from .env file
load_dotenv(override=True)

# === [BOOT CHECKS / SECRETS] ===
def _check_secrets_once():
    """
    Print a single startup warning if required secrets are missing or partial.
    """
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    has_push_token = bool(os.getenv("PUSHOVER_TOKEN"))
    has_push_user  = bool(os.getenv("PUSHOVER_USER"))
    if has_push_token ^ has_push_user:
        # Only one of the two is defined
        print("[boot] WARNING: Partial Pushover config (define both PUSHOVER_TOKEN and PUSHOVER_USER).", flush=True)
    elif not has_push_token and not has_push_user:
        print("[boot] Pushover disabled (no PUSHOVER_*).", flush=True)

    if missing:
        print(f"[boot] WARNING: Missing required secrets: {', '.join(missing)}", flush=True)

# === [RAG CONFIG] ===
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  
_KB_CACHE_PATH = Path("me/.kb_cache.npz")

# === [PUSH NOTIFICATIONS CONFIG] ===
_LAST_PUSH_TS = 0.0
_MIN_PUSH_INTERVAL_SEC = 120  # 2 minutes, can be changed
_SENT_EMAIL_TOKENS = set()

# --- [Typing UX config] ---
TYPING_CHARS_PER_TICK = 3     
TYPING_DELAY_S = 0.045      
# TYPING_CHARS_PER_TICK = 10
# TYPING_DELAY_S = 0.5 
THINKING_TEXT = "Thinking..."

# === [RAG HELPERS] === 
def _chunk_text(text, target_size=900, overlap=200): 
    """
    Split `text` into sentence-aware chunks with overlap, to preserve context.
    Args:
        text (str): The input string to split.
        target_size (int): Target size of each chunk.
        overlap (int): Number of characters to overlap between chunks.
    Returns:
        list[str]: List of chunks.
    """
    text = (text or "").strip()
    if not text:
        return []
    
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # If the sentence is too long, split it into smaller pieces
        if len(s) > target_size:
            for i in range(0, len(s), target_size):
                piece = s[i:i+target_size]
                if cur:
                    if len(cur) + 1 + len(piece) <= target_size:
                        cur = (cur + " " + piece).strip()
                    else:
                        chunks.append(cur)
                        # overlap
                        cur = (cur[-overlap:] + " " + piece).strip() if overlap and len(cur) > overlap else piece
                else:
                    cur = piece
        else:
            # Regular sentence
            if not cur:
                cur = s
            elif len(cur) + 1 + len(s) <= target_size:
                cur = (cur + " " + s).strip()
            else:
                chunks.append(cur)
                cur = (cur[-overlap:] + " " + s).strip() if overlap and len(cur) > overlap else s
    if cur:
        chunks.append(cur)
    return chunks

def _embed_texts(client, texts):
    """
    Compute OpenAI embeddings for a batch of texts.
    Args:
        client (OpenAI): An initialized OpenAI client.
        texts (list[str]): Texts to embed.
    Returns:
        numpy.ndarray: Array of shape (N, D) with float32 embeddings.
    """
    vecs = client.embeddings.create(model=EMBED_MODEL, input=texts).data
    return np.array([v.embedding for v in vecs], dtype=np.float32)

def _cosine_sim(a, b):
    """
    Compute cosine similarity between each row in `a` and a single vector `b`.
    Args:
        a (numpy.ndarray): Matrix of shape (N, D) (embeddings).
        b (numpy.ndarray): Vector of shape (D,) (query embedding).
    Returns:
        numpy.ndarray: Similarity scores of shape (N,).
    """
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return (a @ b)

def _build_kb(openai_client):
    """
    Build an in-memory KB from local files with smart chunking + embedding cache.
    Cache key = SHA256(manifest(paths, mtime, size), EMBED_MODEL)
    Cache file: me/.kb_cache.npz
    Sources:
      - me/summary.txt
      - me/linkedin.pdf (text-extracted)
      - me/knowledge/**/*.md|txt
    Args:
        openai_client (OpenAI): Client used to create embeddings.
    Returns:
        dict: {
            "chunks": list[str],       # text chunks
            "sources": list[str],      # source filename per chunk
            "embeddings": numpy.ndarray|None  # (N, D) or None if empty
        }
    """
    def _list_source_files():
        files = []
        # summary
        p = Path("me/summary.txt")
        if p.exists():
            files.append(p)
        # linkedin
        p = Path("me/linkedin.pdf")
        if p.exists():
            files.append(p)
        # knowledge/*
        kb_dir = Path("me/knowledge")
        if kb_dir.exists():
            for q in kb_dir.glob("**/*"):
                if q.is_file() and q.suffix.lower() in {".md", ".txt"}:
                    files.append(q)
        return files

    def _manifest(files):
        # (path, mtime_ns, size)
        m = [(str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files]
        m.sort()
        return m

    def _manifest_hash(m):
        h = hashlib.sha256()
        h.update(EMBED_MODEL.encode("utf-8"))
        h.update(repr(m).encode("utf-8"))
        return h.hexdigest()

    files = _list_source_files()
    m = _manifest(files)
    mh = _manifest_hash(m)

    # ---- Try cache ----
    if _KB_CACHE_PATH.exists():
        try:
            z = np.load(_KB_CACHE_PATH, allow_pickle=True)
            if z.get("manifest_hash") and str(z["manifest_hash"]) == mh:
                chunks = list(z["chunks"])
                sources = list(z["sources"])
                embeddings = z["embeddings"].astype(np.float32)
                print(f"[kb] cache hit ({_KB_CACHE_PATH}); chunks={len(chunks)}", flush=True)
                return {"chunks": chunks, "sources": sources, "embeddings": embeddings}
            else:
                print("[kb] cache miss (manifest changed) -> rebuild", flush=True)
        except Exception as e:
            print(f"[kb] cache load error -> rebuild: {e}", flush=True)

    # ---- Build from files ----
    docs = []
    # summary
    if Path("me/summary.txt").exists():
        print("[kb] add summary.txt", flush=True)
        docs.append(("summary.txt", Path("me/summary.txt").read_text(encoding="utf-8", errors="ignore")))
    # linkedin
    if Path("me/linkedin.pdf").exists():
        print("[kb] add linkedin.pdf", flush=True)
        try:
            txt = ""
            for p in PdfReader("me/linkedin.pdf").pages:
                txt += (p.extract_text() or "")
            docs.append(("linkedin.pdf", txt))
        except Exception as e:
            print("[kb] pdf read error:", e, flush=True)
    # knowledge/*
    kb_dir = Path("me/knowledge")
    if kb_dir.exists():
        print("[kb] scanning:", kb_dir.resolve(), flush=True)
        for p in kb_dir.glob("**/*"):
            if p.suffix.lower() in {".md", ".txt"}:
                print("[kb] add", p, flush=True)
                docs.append((str(p), p.read_text(encoding="utf-8", errors="ignore")))
    else:
        print("[kb] dir missing:", kb_dir.resolve(), flush=True)

    if not docs:
        print("[kb] no docs found", flush=True)
        return {"chunks": [], "sources": [], "embeddings": None}

    sources, chunks = [], []
    for name, text in docs:
        for ch in _chunk_text(text, target_size=900, overlap=200):
            sources.append(name)
            chunks.append(ch)

    embeddings = _embed_texts(openai_client, chunks)

    # ---- Save cache ----
    try:
        np.savez_compressed(_KB_CACHE_PATH, manifest_hash=np.array(mh),
                            chunks=np.array(chunks, dtype=object),
                            sources=np.array(sources, dtype=object),
                            embeddings=embeddings)
        print(f"[kb] cache saved -> {_KB_CACHE_PATH}", flush=True)
    except Exception as e:
        print(f"[kb] cache save error: {e}", flush=True)

    return {"chunks": chunks, "sources": sources, "embeddings": embeddings}

def search_kb(kb, client, query, k=3): 
    """
    Retrieve top-k most similar KB chunks to a query using cosine similarity.
    Args:
        kb (dict): KB dict returned by `_build_kb`.
        client (OpenAI): OpenAI client for the query embedding.
        query (str): Natural-language question to search.
        k (int): Number of passages to return.
    Returns:
        list[dict]: Each item has {"source": str, "text": str, "score": float}.
    """
    if not kb.get("chunks"):
        return []
    qv = _embed_texts(client, [query])[0]
    sims = _cosine_sim(kb["embeddings"], qv)
    idx = np.argsort(-sims)[:k]
    return [{"source": kb["sources"][i], "text": kb["chunks"][i], "score": float(sims[i])} for i in idx]

def _looks_like_noinfo(text: str) -> bool:
    """
    Heuristic: detect 'I don't know / no info' style replies.
    Used to decide whether to trigger DB fail-safe.
    """
    if not text:
        return True
    t = text.lower()
    patterns = [
    "i don't have", "i do not have", "i don't currently have",
    "i don't know", "i do not know", "not sure", "no information",
    "can't answer", "cannot answer", "i'm not certain",
    ]
    return any(p in t for p in patterns)

# === [SQL Q&A DB HELPER] ===
def _db():
    """
    Open a SQLite connection and ensure required tables exist.
    Tables:
      - qa(question TEXT UNIQUE, answer TEXT, created_at TEXT)
      - evals(ts TEXT, question TEXT, reply TEXT, score REAL, notes TEXT)
    """
    conn = sqlite3.connect("qa.db", timeout=30, isolation_level=None, check_same_thread=False)
    print("[db] using:", os.path.abspath("qa.db"), flush=True)
    # Set up SQLite optimizations
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    # --- Ensure tables exist ---
    # QA table:
    # Stores all user questions and corresponding answers.
    # Columns:
    # - question (TEXT UNIQUE): the normalized user question (unique key)
    # - answer   (TEXT): the assistant's answer (or placeholder if missing)
    # - created_at (TEXT): ISO timestamp when this entry was last updated
    # - status   (TEXT): "unknown" (no answer yet), "answered", or "pending"
    # - lang     (TEXT): detected language of the Q/A (e.g., "he", "en", "other")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa (
            question   TEXT UNIQUE,
            answer     TEXT,
            created_at TEXT,
            status     TEXT,  -- unknown | answered | pending
            lang       TEXT   -- he | en | other
        )
    """)
    # Evals table:
    # Stores evaluation results for assistant replies (quality control).
    # Columns:
    # - ts       (TEXT): ISO UTC timestamp when the evaluation was performed
    # - question (TEXT): the original user question being evaluated
    # - reply    (TEXT): the assistant's reply that was evaluated
    # - faith    (REAL): faithfulness score (1.0â€“5.0) - grounded, no hallucinations
    # - help     (REAL): helpfulness score (1.0â€“5.0) - useful, complete, actionable
    # - clarity  (REAL): clarity score (1.0â€“5.0) - readability, organization
    # - overall  (REAL): overall/average score (1.0â€“5.0)
    # - notes    (TEXT): free-text notes (e.g., "auto-rewritten", evaluator feedback)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evals (
            ts       TEXT,
            question TEXT,
            reply    TEXT,
            faith    REAL,
            help     REAL,
            clarity  REAL,
            overall  REAL,
            notes    TEXT
        )
    """)

    return conn

def _normalize_question(q: str) -> str:
    """Trim and collapse whitespace to avoid duplicate DB rows for the same question."""
    return " ".join((q or "").strip().split())

def _is_placeholder(ans) -> bool:
    """
    Return True if answer is effectively 'unknown' placeholder.
    Treats None/empty and common placeholders as missing.
    """
    if ans is None:
        return True
    s = str(ans).strip().lower()
    return (s == "" or s in {"[tbd]", "tbd", "[todo]", "n/a", "na"})

def db_get_answer(question: str):
    """
    Fetch an existing answer for a given question.
    Returns: {"answer": <str or None>}
    """
    qn = _normalize_question(question)
    print(f"[db] get answer for: {qn!r}", flush=True)
    conn = _db()
    cur = conn.execute("SELECT answer FROM qa WHERE question = ?", (qn,))
    row = cur.fetchone()
    conn.close()
    print("[db] found:", bool(row and row[0]), flush=True)
    return {"answer": row[0]} if row and row[0] is not None else {"answer": None}

def db_save_qa(question: str, answer: str = ""):
    """
    Save or upsert (question, answer) into the qa table, with status/lang.
    - status: 'answered' if non-placeholder answer; else 'unknown'
    - lang: 'he' if contains Hebrew letters, 'en' if ASCII-ish; else 'other'
    Returns: {"saved": "ok"}
    """
    qn = _normalize_question(question)
    placeholder = (answer or "[TBD]").strip()

    # Determine status
    status = "answered"
    if _is_placeholder(placeholder):
        status = "unknown"

    # Simple language detection: Hebrew if contains Hebrew letters
    s = qn + " " + placeholder
    if re.search(r'[\u0590-\u05FF]', s):
        lang = "he"
    elif s.isascii():
        lang = "en"
    else:
        lang = "other"

    print(f"[db] save QA: {qn!r} -> {placeholder!r} | status={status} | lang={lang}", flush=True)
    conn = _db()
    conn.execute(
        "INSERT INTO qa(question, answer, created_at, status, lang) VALUES(?,?,?,?,?) "
        "ON CONFLICT(question) DO UPDATE SET "
        "answer=excluded.answer, created_at=excluded.created_at, status=excluded.status, lang=excluded.lang",
        (qn, placeholder, datetime.now(timezone.utc).isoformat(), status, lang)
    )
    conn.commit()
    conn.close()
    print("[db] saved ok", flush=True)
    return {"saved": "ok"}

def db_save_eval(question: str, reply: str, faith: float, help: float, clarity: float, overall: float, notes: str = ""):
    """
    Keep one evaluation row in the 'evals' table with all grades and comments.
    Returns: {"saved": "ok"}
    """
    conn = _db()
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO evals(ts, question, reply, faith, help, clarity, overall, notes) VALUES(?,?,?,?,?,?,?,?)",
        (ts, question, reply, float(faith), float(help), float(clarity), float(overall), notes or "")
    )
    conn.commit()
    conn.close()
    print(f"[evals] saved: overall={overall:.2f} (F={faith:.1f}/H={help:.1f}/C={clarity:.1f})", flush=True)
    return {"saved": "ok"}

# === [PUSH NOTIFICATIONS] ===
def push(text, priority: str = "normal"):
    """
    Send a Pushover notification with simple rate-limiting (debounce).
    priority:
      - "normal" -> respect rate limit
      - "high"   -> bypass rate limit (used for critical signals like new user email)
    Returns: {"sent": bool, "reason"?: str}
    """
    global _LAST_PUSH_TS
    token = os.getenv("PUSHOVER_TOKEN")
    user  = os.getenv("PUSHOVER_USER")
    if not token or not user:
        print("[push] skipped: missing Pushover credentials", flush=True)
        return {"sent": False, "reason": "missing Pushover credentials"}

    now = time.time()
    if priority != "high":
        if now - _LAST_PUSH_TS < _MIN_PUSH_INTERVAL_SEC:
            print("[push] skipped: rate_limited", flush=True)
            return {"sent": False, "reason": "rate_limited"}

    try:
        r = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "message": text},
            timeout=10
        )
        if r.ok:
            _LAST_PUSH_TS = now
            print("[push] sent ok", flush=True)
            return {"sent": True}
        else:
            print(f"[push] http error: {r.status_code}", flush=True)
            return {"sent": False, "reason": f"http {r.status_code}"}
    except Exception as e:
        print(f"[push] error: {e}", flush=True)
        return {"sent": False, "reason": str(e)}

# === [TOOLS] ===
def record_user_details(email, name="Name not provided", notes="not provided"):
    """
    Records user's contact details and sends a high-priority push.
    Includes a simple in-process dedupe so the same email won't be pushed twice per run.
    """
    token = f"{email.strip().lower()}::{name.strip().lower()}"
    if token in _SENT_EMAIL_TOKENS:
        print("[tool:record_user_details] deduped (already sent for this email/name)", flush=True)
        return {"recorded": "ok", "deduped": True}

    msg = f"[Contact] name={name} | email={email}"
    if notes:
        msg += f" | notes={notes}"
    res = push(msg, priority="high")  # bypass rate-limit for contact details
    _SENT_EMAIL_TOKENS.add(token)
    print(f"[tool:record_user_details] push result: {res}", flush=True)
    return {"recorded": "ok", "push": res}

def record_unknown_question(question):
    """
    Log a question the assistant could not answer and notify via Pushover.
    """
    push(f"Recording {question}")
    return {"recorded": "ok"}

# === [JSON SCHEMA TOOLS] ===
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "format": "email",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

search_knowledge_json = {
    "name": "search_knowledge",
    "description": "Search the user's personal knowledge base for relevant context to answer questions about them.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
            "k": {
                "type": "integer", 
                "description": "How many passages to retrieve", 
                "default": 5,
                "minimum": 1,
                "maximum": 8
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

db_get_answer_json = {
    "name": "db_get_answer",
    "description": "Get an existing answer from the local Q&A database for the exact question text.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The exact user question to look up"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

db_save_qa_json = {
    "name": "db_save_qa",
    "description": "Save (question, answer) into the local Q&A database. Use an empty answer if not yet known.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The exact user question to store"},
            "answer":   {"type": "string", "description": "The answer text; can be empty to store a placeholder", "default": ""}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
    {"type": "function", "function": search_knowledge_json},
    {"type": "function", "function": db_get_answer_json},
    {"type": "function", "function": db_save_qa_json},
]

# Only allow these server-side tool functions to be invoked by the LLM
ALLOWED_FUNCS = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
    "db_get_answer": db_get_answer,
    "db_save_qa": db_save_qa,
}

# === [EVALUATOR PROMPTS & PAYLOAD BUILDERS] ===
# System prompt for the evaluator that returns compact JSON scores
evaluator_system_prompt = (
    "You are a strict but fair evaluator for an assistant's response.\n"
    "Rate the reply on 3 axes from 1.0 (poor) to 5.0 (excellent):\n"
    "- faithfulness: grounded in provided context without hallucinations\n"
    "- helpfulness: usefulness, completeness, actionability for the user's goal\n"
    "- clarity: organization, readability, language appropriateness\n"
    "Respond ONLY in compact JSON with keys: faith, help, clarity, overall, feedback.\n"
    "overall should be the average of the three, rounded to 2 decimals."
)

# System prompt for rewriting a reply (improve clarity/helpfulness/faithfulness)
rewriter_system_prompt = (
    "You are a rewriting assistant. Improve the given reply without changing factual content.\n"
    "Goals: increase faithfulness to provided context, improve helpfulness and clarity.\n"
    "Constraints: keep the user's language; keep code/identifiers in English; be concise."
)

def evaluator_user_prompt(reply: str, message: str, history: list):
    """
    Build a compact payload for the evaluator to keep tokens low and signal context clearly.
    """
    return {
        "reply": reply,
        "user_message": message,
        # Keep only a small slice of history to avoid long prompts
        "history": history[-6:] if history else []
    }

def rewriter_user_prompt(reply: str, message: str, feedback: str = ""):
    """
    Build a compact payload for the rewriter model.
    """
    return {
        "user_message": message,
        "feedback": feedback or "Improve clarity, structure, and ensure it is grounded in the given context.",
        "reply_to_rewrite": reply
    }

# === [EVALUATOR MODELS] ===
class Evaluation(BaseModel):
    """
    Minimal outcome for gating: is the reply acceptable to show as-is?
    Also carry feedback for possible rewrite.
    """
    is_acceptable: bool
    feedback: str

class EvaluationScores(BaseModel):
    """
    Full score breakdown for analytics & DB storage.
    """
    faith: float
    help: float
    clarity: float
    overall: float

class Me:

    def __init__(self):
        self.openai = OpenAI().with_options(timeout=30)
        self.name = "Yuval Zaafrani"
        try:
            if Path("me/summary.txt").exists():
                with open("me/summary.txt", "r", encoding="utf-8") as f:
                    self.summary = f.read()
            else:
                print("[boot] summary.txt missing; continuing with empty summary", flush=True)
                self.summary = ""
        except Exception as e:
            print(f"[boot] summary.txt read error: {e}", flush=True)
            self.summary = ""

        self.kb = _build_kb(self.openai)
        self.linkedin = ""
        try:
            if self.kb and self.kb.get("chunks") and self.kb.get("sources"):
                li_chunks = [
                    ch for ch, src in zip(self.kb["chunks"], self.kb["sources"])
                    if str(src).endswith("linkedin.pdf")
                ]
                self.linkedin = "".join(li_chunks)
            if not self.linkedin:
                print("[boot] linkedin text empty; continuing with empty profile", flush=True)
        except Exception as e:
            print(f"[boot] linkedin aggregation error: {e}", flush=True)
            self.linkedin = ""
        self._unknown_logged = False

        print("[boot] CWD:", os.getcwd(), flush=True)
        print("[boot] KB chunks:", len(self.kb.get("chunks") or []), flush=True)
        print("[boot] KB sources (unique):", len(set(self.kb.get("sources") or [])), flush=True)
        print("[boot] summary chars:", len(getattr(self, "summary", "") or ""), flush=True)
        print("[boot] linkedin chars:", len(getattr(self, "linkedin", "") or ""), flush=True)

    def search_knowledge(self, query: str, k: int = 5):
        """
        Instance-bound KB search.
        Returns: {"matches": str, "sources": list[str]}
        """
        if not getattr(self, "kb", None) or not (self.kb.get("chunks") or []):
            print("[rag] KB empty; returning no matches", flush=True)
            return {"matches": "", "sources": []}

        def _run(q, kk):
            results = search_kb(self.kb, self.openai, q, k=kk)
            print(f"[rag] query={q!r} hits={len(results)}", flush=True)
            for i, r in enumerate(results):
                snippet = (r["text"][:100] + "...") if len(r["text"]) > 100 else r["text"]
                print(f"[rag]  {i+1}. {r['source']} | score={r['score']:.3f} | {snippet}", flush=True)
            return results

        results = _run(query, k)
        max_score = max((r["score"] for r in results), default=0.0)

        if max_score < 0.45:
            alt_queries = [
                query,
                f"{query} projects",
                f"{query} from projects.md",
                "list of projects",
                "project case study",
                "portfolio projects",
                "projects yuval zaafrani",
            ]
            merged = {}
            for q in alt_queries:
                for r in _run(q, k):
                    key = (r["source"], r["text"])
                    merged[key] = max(merged.get(key, 0.0), r["score"])

            results = [{"source": s, "text": t, "score": sc} for (s, t), sc in merged.items()]
            results.sort(key=lambda x: -x["score"])
            results = results[:k]
            print(f"[rag] merged hits={len(results)} (after expansion)", flush=True)

        joined = "\n\n".join([f"[{r['source']}] {r['text']}" for r in results])
        if len(joined) > 4000:
            joined = joined[:4000] + "\n...[truncated]"
        print("[rag] joined preview:", joined[:200].replace("\n", " ") + ("..." if len(joined) > 200 else ""), flush=True)

        sources = [r["source"] for r in results]
        return {"matches": joined, "sources": list(dict.fromkeys(sources))}

    def handle_tool_call(self, tool_calls):
        """
        Execute a list of tool calls returned by the LLM.
        Also: if record_unknown_question is called, we auto-save the question to DB.
        Args:
            tool_calls (list): Tool call objects with .function.name and JSON arguments.
        Returns:
            list[dict]: Messages with role="tool" to append back into the chat transcript.
        """
        results = []
        # ---------- UI hook : allow UI to disable RAG (search_knowledge) ----------
        # The UI sets `self._ui_disable_kb = True/False` before calling me.chat().
        # If this flag is True, we will *stub* search_knowledge tool results so the model
        # won't use KB for this turn (other tools still work as usual).
        ui_disable_kb = getattr(self, "_ui_disable_kb", False)
        # -------------------------------------------------------------------------------
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                print(f"[tool] {tool_name} args={arguments}", flush=True)

                if tool_name == "search_knowledge":
                    if ui_disable_kb:
                        # Stub out KB when toggle is OFF
                        result = {"matches": "", "sources": []}
                        # Do NOT cache _last_sources here (it's intentionally empty)
                        self._last_sources = []
                    else:
                        # Normal KB flow 
                        result = self.search_knowledge(arguments["query"], arguments.get("k", 5))
                        try:
                            # Cache the sources for the next call 
                            self._last_sources = list(result.get("sources") or [])
                        except Exception:
                            self._last_sources = []
                else:
                    tool = ALLOWED_FUNCS.get(tool_name)
                    if tool is None:
                        result = {"error": f"tool '{tool_name}' is not allowed"}
                    else:
                        result = tool(**arguments)

                if tool_name == "record_unknown_question" and "question" in arguments:
                    try:
                        db_save_qa(arguments["question"], "")
                        self._unknown_logged = True
                    except Exception as e:
                        print(f"[warn] auto db_save_qa failed: {e}", flush=True)

                results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})

            except Exception as e:
                err = {"error": f"tool '{tool_name}' failed", "details": str(e)}
                print("[tool-error]", err, flush=True)
                results.append({"role": "tool", "content": json.dumps(err, ensure_ascii=False), "tool_call_id": getattr(tool_call, "id", None)})

        return results
    
    def system_prompt(self):
        """
        Build the system prompt that enforces grounding, tool policy, and style.
        Returns:
            str: The full system prompt including summary and LinkedIn context.
        """
        system_prompt = (
            # Identity & core rule
            f"You are {self.name}, responding on {self.name}'s professional website. "
            f"Your #1 rule: DO NOT HALLUCINATE. Only answer with facts grounded in the provided context "
            f"(Professional Summary, LinkedIn) or retrieved via tools (search_knowledge, db_get_answer). "
            f"If the answer is not explicitly supported by those sources, you MUST NOT guess. "
            f"Instead, use record_unknown_question to log the gap so it can be filled later.\n\n"

            # Style & audience
            f"Tone: professional, concise, and helpful. Default to the user's language; if unclear, use English. "
            f"When the user mixes Hebrew and English, prefer full sentences in Hebrew and keep code/identifiers in English. "
            f"Prefer short, direct answers; use bullet points when helpful.\n\n"

            # Tool policy (strict order & conditions)
            f"Tool policy:\n"
            f"1) For any question about {self.name}'s background, projects, skills, achievements, or availability, "
            f"   first call search_knowledge with a focused query. Use the results to ground your answer.\n"
            f"   If search_knowledge returns any matches (>0), you MUST use them to answer. "
            f"   Do NOT call db_get_answer unless search_knowledge returned 0 results.\n"
            f"2) If search_knowledge (and the provided Summary/LinkedIn) still don't contain the needed facts, "
            f"   call db_get_answer(question). If no answer is found, call record_unknown_question with the user's question (verbatim), "
            f"   then call db_save_qa(question, answer=\"\") to store it for later completion. "
            f"   After that, reply briefly that you don't currently have that info.\n"
            f"3) If the user shares an email or asks to be contacted, call record_user_details with the email (and name/notes if given). "
            f"   Never invent or infer an email/name.\n"
            f"4) If the user asks a repeating FAQ item, first try db_get_answer(question). If it exists, use it verbatim; "
            f"   otherwise proceed as in step 2.\n\n"

            # Grounding rules
            f"Grounding rules (strict):\n"
            f"- Every factual claim about {self.name} must be traceable to the provided Summary/LinkedIn or search_knowledge/db_get_answer results.\n"
            f"- Do not infer dates, employers, titles, metrics, or tech stacks unless explicitly present in sources.\n"
            f"- If sources partially cover the question, answer only the supported part and state what is unknown.\n"
            f"- Never cite private or sensitive info unless the user provided it in this conversation.\n\n"

            # Output format & Sources
            f"Output format:\n"
            f"- Provide the answer first.\n"
            f"- When you used retrieved passages, add a short 'Sources:' line listing only the filenames "
            f"(e.g., projects.md, skills.md, achievements.md, summary.txt, linkedin.pdf). "
            f"  Use the 'sources' array returned by search_knowledge when present. "
            f"  Do NOT include raw embeddings or long quotes; keep it succinct.\n\n"

            # Engagement
            f"If the conversation becomes more interactive, you may (politely) offer to follow up by email and, if they consent and provide it, "
            f"use record_user_details. Do not pressure the user.\n\n"

            # Provided context as ground truth
            f"Below is {self.name}'s current context. Use it as ground truth:\n\n"
            f"## Professional Summary:\n{self.summary}\n\n"
            f"## LinkedIn Profile:\n{self.linkedin}\n\n"
            f"Always remain in character as {self.name}, be accurate, and prefer 'I don't have that info yet' over guessing."
        )
        return system_prompt

    def _chat_with_backoff(self, messages, tools, model="gpt-4o-mini", max_retries=3):
        """
        Call OpenAI Chat Completions with simple exponential backoff + jitter.
        Retries on common transient errors (timeouts / rate limits / connection errors).
        """
        attempt = 0
        while True:
            try:
                return self.openai.chat.completions.create(model=model, messages=messages, tools=tools)
            except Exception as e:
                attempt += 1
                # classify error: try again on timeouts/ratelimit/connection
                err = str(e).lower()
                transient = any(k in err for k in [
                    "rate limit", "ratelimit", "timeout", "timed out",
                    "temporary", "temporarily", "connection", "connect",
                    "service unavailable", "503", "429"
                ])
                if attempt > max_retries or not transient:
                    print(f"[openai] giving up after {attempt} attempts: {e}", flush=True)
                    raise
                # backoff with jitter
                sleep_s = min(8, (2 ** (attempt - 1))) + random.random()
                print(f"[openai] transient error: {e} -> retry {attempt}/{max_retries} in ~{sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)

    def evaluate_reply(self, reply: str, message: str, history: list, accept_threshold: float = 3.5, max_retries: int = 2): 
        """
        Evaluate a reply and return:
          - scores: EvaluationScores (faith/help/clarity/overall)
          - gate:   Evaluation (is_acceptable + feedback)
        The evaluator MUST return compact JSON; robust parsing is applied.
        """
        user_payload = evaluator_user_prompt(reply, message, history)
        messages = [
            {"role": "system", "content": evaluator_system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ]

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._chat_with_backoff(messages, tools=[], model="gpt-4o-mini", max_retries=3)
                text = (resp.choices[0].message.content or "").strip()

                # Extract JSON if wrapped
                m = re.search(r'\{.*\}', text, re.DOTALL)
                raw = m.group(0) if m else text
                data = json.loads(raw)

                faith   = float(data.get("faith", 0.0))
                help_   = float(data.get("help", 0.0))
                clarity = float(data.get("clarity", 0.0))
                overall = float(data.get("overall", (faith + help_ + clarity) / 3.0))
                feedback = str(data.get("feedback", "")).strip()

                scores = EvaluationScores(faith=faith, help=help_, clarity=clarity, overall=round(overall, 2))
                gate = Evaluation(is_acceptable=(scores.overall >= accept_threshold), feedback=feedback)
                return scores, gate
            except Exception as e:
                print(f"[eval] parse error (attempt {attempt}): {e}", flush=True)
                if attempt == max_retries:
                    # Conservative fallback: neutral scores + acceptable
                    scores = EvaluationScores(faith=3.0, help=3.0, clarity=3.0, overall=3.0)
                    gate = Evaluation(is_acceptable=True, feedback="fallback")
                    return scores, gate

    def revise_reply(self, reply: str, message: str, feedback: str = ""): 
        """
        Rewrite a reply for better faithfulness/helpfulness/clarity without adding new facts.
        Constraints:
          - Keep user's language (Heb/Eng) and keep code/identifiers in English
          - Be concise and well-structured
        """
        payload = rewriter_user_prompt(reply, message, feedback)
        messages = [
            {"role": "system", "content": rewriter_system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        resp = self._chat_with_backoff(messages, tools=[], model="gpt-4o-mini", max_retries=3)
        new_text = (resp.choices[0].message.content or "").strip()
        return new_text
    
    def chat(self, message, history):
        """
        Gradio chat handler + smart DB fail-safe at the end.
        Args:
            message (str): Latest user message.
            history (list[dict]): Prior messages in {"role","content"} format.
        Returns:
            str: Final assistant message content to render in the UI.
        """
        self._unknown_logged = False
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        # hard cap to avoid infinite tool loops
        max_turns = 8  
        turns = 0

        while not done and turns < max_turns:
            response = self._chat_with_backoff(messages, tools, model="gpt-4o-mini", max_retries=3)
            if response.choices[0].finish_reason == "tool_calls":
                msg = response.choices[0].message
                results = self.handle_tool_call(msg.tool_calls)
                messages.append(msg)
                messages.extend(results)
                turns += 1
            else:
                done = True
        # Graceful fallback to the user if we hit the cap
        if not done:
            print("[guard] max_turns reached; breaking out of tool loop", flush=True)
            qn = _normalize_question(message)
            if not getattr(self, "_unknown_logged", False):
                try:
                    record_unknown_question(message)
                    db_save_qa(qn, "")
                    self._unknown_logged = True
                except Exception as e:
                    print(f"[warn] fallback db log failed: {e}", flush=True)
            return (
                "Sorry, I couldn't complete this request right now (too many tool steps). "
                "I've logged your question so it can be addressed soon."
            )

        final_text = response.choices[0].message.content or ""

        # Fail-safe
        if _looks_like_noinfo(final_text):
            print("Fail-safe (post-loop): consulting DB and logging gap", flush=True)
            qn = _normalize_question(message)
            got = db_get_answer(qn)
            ans = (got or {}).get("answer")

            if ans and not _is_placeholder(ans):
                return ans

            if not getattr(self, "_unknown_logged", False):
                record_unknown_question(message)
                db_save_qa(qn, "")
                self._unknown_logged = True
            return (
                "I don't currently have that information. I've logged your question so it can be addressed soon.\n"
                "In the meantime, you can ask about my projects, skills, or achievements."
            )

        # --- [QUALITY EVALUATION & OPTIONAL REVISION] ---
        try:
            scores, gate = self.evaluate_reply(final_text, message, history, accept_threshold=3.5)
            improved = False

            if not gate.is_acceptable:
                print(f"[eval] low score {scores.overall:.2f} -> revising reply", flush=True)
                new_text = self.revise_reply(final_text, message, gate.feedback)
                if new_text and len(new_text) > 0:
                    final_text = new_text
                    improved = True

            # Persist evaluation (store the final reply shown to the user)
            db_save_eval(
                question=message,
                reply=final_text,
                faith=scores.faith,
                help=scores.help,
                clarity=scores.clarity,
                overall=scores.overall,
                notes=("auto-rewritten" if improved else (gate.feedback or ""))
            )
        except Exception as e:
            print(f"[eval] evaluator error: {e}", flush=True)

        # Add automatic sources if search_knowledge was called but the model didn't add them itself
        try:
            if getattr(self, "_last_sources", None):
                # Don't add twice if Sources: already exists
                if "Sources:" not in (final_text or ""):
                    short_sources = ", ".join(dict.fromkeys(self._last_sources))  # Unique and preserve order
                    final_text = (final_text or "").rstrip() + f"\n\nSources: {short_sources}"
                # Clear for the next interaction
                self._last_sources = []
        except Exception:
            pass

        return final_text

# --- [Utility functions for UI (streaming, email, metrics)] ---
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

def share_email_action(me: "Me", email: str, name: str):
    """Validate and forward email to the existing record_user_details() tool."""
    email = (email or "").strip()
    name = (name or "").strip() or "Name not provided"
    if not EMAIL_RE.match(email):
        return gr.update(value="Invalid email address.", visible=True)
    # call your existing helper
    res = record_user_details(email=email, name=name)
    ok = res.get("recorded") == "ok"
    return gr.update(value=("Thanks! Sent." if ok else f"{res}"), visible=True)

def add_user_message(msg: str, messages: list[dict] | None):
    """
    Chatbot(type='messages'):
    messages = [{'role': 'user'|'assistant', 'content': str}, ...]
    """
    if not (msg or "").strip():
        return "", (messages or [])
    messages = (messages or []) + [{"role": "user", "content": msg}]
    return "", messages

def add_thinking_placeholder(messages: list[dict] | None): # !!!!
    """
    Insert a visible "Thinking..." assistant bubble immediately.
    This runs as a quick step before the streaming generator to guarantee
    the placeholder shows up on hosted deployments as well.
    """
    messages = (messages or [])
    if messages and messages[-1].get("role") == "assistant":
        # Already has an assistant message (e.g., previous placeholder)
        return messages
    return messages + [{"role": "assistant", "content": THINKING_TEXT}]

def stream_reply_messages(me: "Me", messages: list[dict], use_kb_flag: bool):
    """
    Streams assistant replies for Chatbot(type="messages") with:
      - Immediate "Thinking..." placeholder
      - Simulated typing effect for final answer
    """

    # If no messages exist yet, nothing to stream
    if not messages:
        yield messages
        return

    # Toggle Knowledge Base usage according to the checkbox
    me._ui_disable_kb = not bool(use_kb_flag)

    # Find the last user message index to build clean history even if a
    # placeholder assistant bubble was already appended by a previous step
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if (messages[i] or {}).get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx == -1:
        yield messages
        return
    user_msg = messages[last_user_idx]["content"]
    history_for_me = messages[:last_user_idx]

    # If a previous step already displayed the placeholder, keep it visible;
    # otherwise, display it here as a fallback.
    if not (messages and messages[-1].get("role") == "assistant"):
        yield history_for_me + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": THINKING_TEXT},
        ]
    else:
        # Surface current state (with placeholder) to ensure the client shows it
        yield messages

    # --- STEP 3: Generate the full assistant reply (blocking call) ---
    full = me.chat(user_msg, history_for_me)
    full = full or ""  # Fallback to empty string if None

    # --- STEP 4: Replace "Thinking..." with a typing simulation ---
    # If the reply is very short, show it immediately
    if len(full) <= TYPING_CHARS_PER_TICK:
        yield history_for_me + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": full},
        ]
        return

    # Otherwise, reveal the reply gradually to simulate typing
    partial = ""
    for i in range(0, len(full), TYPING_CHARS_PER_TICK):
        partial = full[: i + TYPING_CHARS_PER_TICK]
        yield history_for_me + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": partial},
        ]
        # Small delay between updates to mimic human typing speed
        try:
            time.sleep(TYPING_DELAY_S)
        except Exception:
            pass

    # --- STEP 5: Ensure the full message is shown at the end ---
    yield history_for_me + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": full},
    ]

def load_css() -> str:
    css_path = Path(__file__).with_name("style.css")
    try:
        css = css_path.read_text(encoding="utf-8")
    except Exception as e:
        print("[boot][CSS] ERROR reading:", css_path, e)
        css = ""
    print(f"[boot][CSS] path={css_path} bytes={len(css)}")
    return css

# --- [CLEAN, MODULAR GRADIO UI (Blocks)] ---
def build_ui_app():
    me = Me()  

    # Load external CSS file
    custom_css = load_css()

    with gr.Blocks(
        title="Career Conversations with Yuval Zaafrani",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:

        gr.HTML(f"<style>{custom_css}</style>", visible=False)

        # Centered page title 
        gr.Markdown("# Career Conversation with Yuval Zaafrani", elem_classes=["page-title"])

        with gr.Row():
            # ---- Left: Main CHAT BOX ----
            with gr.Column(scale=8):
                chat = gr.Chatbot(type="messages", value=[], height=650, label="Chat", elem_id="chatArea")
                # ONE unit: textbox (left) + tiny controls (right)
                with gr.Group(elem_classes=["input-shell"]):
                    with gr.Row(elem_classes=["input-row"]):
                        with gr.Column(elem_classes=["col-grow"]):
                            user_box = gr.Textbox(
                                placeholder="Type a message and press Enter...",
                                label=None, show_label=False, lines=1
                            )
                        with gr.Column(elem_classes=["col-auto", "control-col", "card-xxs"]):
                            send_btn  = gr.Button("Send",  variant="secondary", elem_classes=["btn-primary"], elem_id="sendBtnTest") 
                            clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["btn-outline"]) 
                            use_kb    = gr.Checkbox(value=True, label="Use KB (RAG)")

            # ---- Right: sidebar ----
            with gr.Column(scale=1, elem_classes=["sidebar", "sidebar-compact"]):
                with gr.Group(elem_classes=["sidebar-scroll"]):
                    gr.Markdown("### About me:")

                    about = (me.summary or "").strip() or "No summary file found."
                    if len(about) > 1200:
                        about = about[:1200]
                    try:
                        about = re.sub(r"[Bb]eyo\.{0,3}$", "", about).rstrip()
                    except Exception:
                        pass

                    about += ("\n\n**For more questions, you can ask me here in the chat "
                            "or send details and I will get back to you as soon as possible :)**")
                    gr.Markdown(about, elem_id="aboutCard")

                    # Contact: extra small card + gap
                    with gr.Column(elem_classes=["card", "card-xxs"]):
                        gr.Markdown("### Contact me")
                        name = gr.Textbox(label="Name",  placeholder="Your name", lines=1)
                        email = gr.Textbox(label="Email", placeholder="name@example.com", lines=1)
                        share_btn = gr.Button("Send contact")
                        share_result = gr.Textbox(label="Status", interactive=False, visible=False)

        # ---- Wiring the events (messages API) ----
        def _stream_reply_messages_wrapper(messages: list[dict], use_kb_flag: bool):
            for out in stream_reply_messages(me, messages, use_kb_flag):
                yield out
        
        # ENTER button 
        submit_evt = user_box.submit(
            fn=add_user_message,
            inputs=[user_box, chat],
            outputs=[user_box, chat],
            queue=True,
            show_progress=False,
        )
        # Send button
        click_evt = send_btn.click(
            fn=add_user_message,
            inputs=[user_box, chat],
            outputs=[user_box, chat],
            queue=True,
            show_progress=False,
        )
        # Stream the reply
        submit_evt.then(
            fn=add_thinking_placeholder,
            inputs=[chat],
            outputs=[chat],
            queue=True,
            show_progress=False,
        ).then(
            fn=_stream_reply_messages_wrapper,
            inputs=[chat, use_kb],
            outputs=chat,
            queue=True,
            show_progress=False,
        )
        
        click_evt.then(
            fn=add_thinking_placeholder,
            inputs=[chat],
            outputs=[chat],
            queue=True,
            show_progress=False,
        ).then(
            fn=_stream_reply_messages_wrapper,
            inputs=[chat, use_kb],
            outputs=chat,
            queue=True,
            show_progress=False,
        )

        # Clear chat
        clear_btn.click(
            lambda: (gr.update(value=[]), ""),  
            outputs=[chat, user_box],
            queue=False,
            show_progress=False,
        )

        # Contact button flow with visual feedback
        # 1) Immediately show "Sending..." + disable button (triggers plane animation via CSS)
        start_evt = share_btn.click(
            fn=lambda: (
                gr.update(value="Sending...", interactive=False),
                gr.update(value="Sending...", visible=True),
            ),
            inputs=None,
            outputs=[share_btn, share_result],
            queue=False,
            show_progress=False,
        )
        # 2) Perform the send (server call)
        after_send = start_evt.then(
            fn=lambda n, e: share_email_action(me, e, n),
            inputs=[name, email],
            outputs=[share_result],
            queue=True,
            show_progress=False,
        )
        # 3) Restore button label and enable it again
        after_send.then(
            fn=lambda: gr.update(value="Send contact", interactive=True),
            inputs=None,
            outputs=[share_btn],
            queue=False,
            show_progress=False,
        )
    # enable queue
    demo.queue()    
    return demo

# Expose for local run & deploy
app = build_ui_app()

if __name__ == "__main__":
    app.launch()

    