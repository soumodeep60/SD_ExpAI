# knowledge_ingest.py — ingest local PDFs into Gemini Files API
import os, time
from typing import List
from google import genai

def ingest_local_pdfs(pdf_dir: str, api_key: str):
    """
    Upload every *.pdf in pdf_dir to Gemini Files API and return ACTIVE file handles.
    Safe to call on app start. Skips non-PDFs. Idempotent enough for typical restarts.
    """
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing.")
    if not os.path.isdir(pdf_dir):
        return []

    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()

    handles: List[object] = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        try:
            f = client.files.upload(file=path)
            # Wait for processing → ACTIVE
            while getattr(f, "state", None) and f.state.name == "PROCESSING":
                time.sleep(1.0)
                f = client.files.get(name=f.name)
            if f.state.name == "ACTIVE":
                handles.append(f)
        except Exception as e:
            # Swallow per-file errors; app still boots
            print(f"[knowledge_ingest] Upload failed for {fname}: {e}")
    return handles

def build_cache_from_handles(handles, model: str, api_key: str, ttl_seconds: int = 3600):
    """
    Optional: Create a 1-hour context cache so you don’t resend big docs every turn.
    Returns cache.name or None on failure.
    """
    if not handles:
        return None
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()
    try:
        from google.genai import types
        cache = client.caches.create(
            model=model,
            contents=[types.Content(parts=handles)],
            config=types.CreateCachedContentConfig(ttl_seconds=ttl_seconds),
        )
        return cache.name
    except Exception as e:
        print(f"[knowledge_ingest] Cache creation failed: {e}")
        return None
