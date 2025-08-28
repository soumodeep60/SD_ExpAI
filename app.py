# app.py — Gemini Chat + Excel + PDFs (Streamlit + google-genai)

import os, io, time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Experience insights app by Soumodeep", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .title-font {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 48px !important;
        font-weight: bold !important;
        color: white !important;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="title-font">Experience insights app</p>', unsafe_allow_html=True)
st.write("Powered by Google Gemini")

# ---------- Sidebar: API + model ----------
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("GEMINI_API_KEY", type="password")
model = st.sidebar.selectbox("Model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"], 0)
temperature = 0.2
save_tokens = st.sidebar.checkbox("Disable thinking (faster/cheaper)", value=True)

# ---------- Ensure session keys ----------
if "pdf_cache_name" not in st.session_state:
    st.session_state.pdf_cache_name = None
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []        # user-uploaded Files API handles
if "history" not in st.session_state:
    st.session_state.history = []
if "hidden_files" not in st.session_state:
    st.session_state.hidden_files = []     # handles from local knowledge/

# ---------- Hidden knowledge (local 'knowledge/' folder) — NO UI ----------
KNOWLEDGE_DIR = st.secrets.get("KNOWLEDGE_DIR", "knowledge")  # defaults to ./knowledge
USE_CACHE_DEFAULT = bool(st.secrets.get("GEMINI_USE_CACHE", True))

def _as_parts(handles):
    """Convert Files API handles into Parts Gemini can read."""
    parts = []
    if not handles:
        return parts
    # Use a single client here to avoid re-auth in loop
    os.environ["GEMINI_API_KEY"] = api_key or os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client()
    for h in handles:
        uri = getattr(h, "uri", None)
        mt = getattr(h, "mime_type", None) or "application/pdf"
        if uri:
            parts.append(types.Part.from_uri(file_uri=uri, mime_type=mt))
        else:
            # Try to refresh handle to get URI
            try:
                name = getattr(h, "name", None)
                if name:
                    h2 = client.files.get(name=name)
                    if getattr(h2, "uri", None):
                        parts.append(types.Part.from_uri(file_uri=h2.uri, mime_type=getattr(h2, "mime_type", mt)))
            except Exception:
                pass
    return parts

def _ingest_local_pdfs(pdf_dir: str, api_key: str):
    """Upload every *.pdf in pdf_dir to Gemini Files API and return ACTIVE file handles."""
    if not api_key or not os.path.isdir(pdf_dir):
        return []
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()
    handles = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        try:
            f = client.files.upload(file=path)
            # Wait until ACTIVE
            while getattr(f, "state", None) and f.state.name == "PROCESSING":
                time.sleep(1.0)
                f = client.files.get(name=f.name)
            if f.state.name == "ACTIVE":
                handles.append(f)
        except Exception as e:
            # Don't crash the app if one file fails
            print(f"[knowledge] upload failed for {fname}: {e}")
    return handles

def _build_cache_from_handles(handles, model: str, api_key: str, ttl_seconds: int = 3600):
    """Create a short-lived context cache from local knowledge handles. Returns cache.name or None."""
    if not handles:
        return None
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()
    try:
        file_parts = _as_parts(handles)  # convert to Parts
        if not file_parts:
            return None
        cache = client.caches.create(
            model=model,
            contents=[types.Content(parts=file_parts)],
            config=types.CreateCachedContentConfig(ttl_seconds=ttl_seconds),
        )
        return cache.name
    except Exception as e:
        print(f"[knowledge] cache creation failed: {e}")
        return None

# One-time per cold start: ingest local PDFs and optionally build a cache
if api_key and not st.session_state.hidden_files and os.path.isdir(KNOWLEDGE_DIR):
    st.session_state.hidden_files = _ingest_local_pdfs(KNOWLEDGE_DIR, api_key)
    if USE_CACHE_DEFAULT and st.session_state.hidden_files and not st.session_state.pdf_cache_name:
        st.session_state.pdf_cache_name = _build_cache_from_handles(
            st.session_state.hidden_files, model=model, api_key=api_key, ttl_seconds=3600
        )
# <<< end hidden knowledge block

# ---------- Sidebar: PDF “training” (Files API) ----------
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Knowledge PDFs")
use_cache = st.sidebar.checkbox("Use explicit context cache (for big corpora)")
pdf_uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# ---------- Sidebar: Upload Excel ----------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx / .xls)", type=["xlsx", "xls"])
df = None
if uploaded:
    try:
        df = pd.read_excel(uploaded)  # requires openpyxl
        st.sidebar.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        st.sidebar.dataframe(df.head())
    except Exception as e:
        st.sidebar.error(f"Failed to read Excel: {e}\nTry: pip install openpyxl")

# ---------- Attach mode for Excel ----------
attach_mode = st.sidebar.radio(
    "Attach to prompt",
    ["Sample rows (compact)", "Stat summary (describe + groupby)"],
    index=0,
)
# optional groupby selector for summary
group_col = "<none>"
if attach_mode.startswith("Stat"):
    if df is not None:
        group_col = st.sidebar.selectbox("Group by (optional)", ["<none>"] + list(df.columns), 0)
max_rows = 1000000

# ---------- Controls ----------
if st.sidebar.button("🧹 Clear chat"):
    st.session_state.history = []
    st.rerun()

# ---------- Chat history ----------
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

def add_and_render(role: str, text: str):
    st.session_state.history.append((role, text))
    with st.chat_message(role):
        st.markdown(text)

def build_data_context(df: pd.DataFrame) -> str:
    """Return a compact, model-friendly text context based on the chosen attach_mode."""
    if df is None or df.empty:
        return ""
    if attach_mode.startswith("Sample"):
        sample = df.head(max_rows)
        buf = io.StringIO()
        sample.to_csv(buf, index=False)
        schema = "\n".join(f"- {c}: {df[c].dtype}" for c in df.columns)
        return (
            "DATA MODE: SAMPLE\n"
            "Columns and inferred dtypes:\n" + schema +
            "\n\nCSV sample (header included):\n" + buf.getvalue()
        )
    else:
        pieces = []
        pieces.append("DATA MODE: SUMMARY")
        pieces.append("describe():")
        with pd.option_context("display.max_colwidth", 200, "display.width", 1000):
            pieces.append(df.describe(include="all", datetime_is_numeric=True).to_string())
        if group_col and group_col != "<none>":
            try:
                gb = df.groupby(group_col).agg(["count", "mean", "sum", "min", "max"]).fillna("")
                pieces.append(f"\nGroupby '{group_col}' (count/mean/sum/min/max):")
                pieces.append(gb.to_string())
            except Exception as e:
                pieces.append(f"\n[Groupby failed on '{group_col}': {e}]")
        return "\n".join(pieces)

def to_gemini_messages(user_prompt: str):
    """Convert history + data context + new question to Gemini content dicts, including PDFs."""
    msgs = []
    # past history
    for role, text in st.session_state.history:
        msgs.append({"role": "user" if role == "user" else "model", "parts": [{"text": text}]})

    # assemble current user turn
    parts = []

    # include cached content reference if present (explicit context cache)
    if use_cache and st.session_state.pdf_cache_name:
        parts.append(types.CachedContentRef(name=st.session_state.pdf_cache_name))
    else:
        # include hidden knowledge files (from local 'knowledge/') as Parts
        if st.session_state.get("hidden_files"):
            parts.extend(_as_parts(st.session_state.hidden_files))
        # include uploaded PDF files (live Files API handles) as Parts
        if st.session_state.pdf_files:
            parts.extend(_as_parts(st.session_state.pdf_files))

    # include Excel-derived context if any
    if df is not None:
        parts.append({"text": build_data_context(df)})

    # finally the user's question
    if user_prompt:
        parts.append({"text": f"QUESTION:\n{user_prompt}"})

    if parts:
        msgs.append({"role": "user", "parts": parts})
    return msgs

# ---------- Upload PDFs (via Files API) ----------
def ensure_client():
    os.environ["GEMINI_API_KEY"] = api_key
    return genai.Client()

if pdf_uploads and api_key:
    client = ensure_client()
    new_handles = []
    with st.sidebar.status("Uploading & processing PDFs…", expanded=True) as status:
        for f in pdf_uploads:
            try:
                file_obj = client.files.upload(file=f)  # supports PDF; stored ~48h
                # wait for processing -> ACTIVE
                while getattr(file_obj, "state", None) and file_obj.state.name == "PROCESSING":
                    time.sleep(1.2)
                    file_obj = client.files.get(name=file_obj.name)
                if file_obj.state.name != "ACTIVE":
                    st.sidebar.warning(f"{f.name}: state={file_obj.state.name}")
                else:
                    st.sidebar.write(f"✓ {f.name} ready")
                    new_handles.append(file_obj)
            except Exception as e:
                st.sidebar.error(f"{f.name}: upload failed — {e}")
        st.session_state.pdf_files.extend(new_handles)

# ---------- Create/refresh explicit cache from the uploaded PDFs ----------
if use_cache and api_key and st.button("⚡ Build / Refresh cache from PDFs"):
    if not st.session_state.pdf_files:
        st.warning("Upload at least one PDF first.")
    else:
        client = ensure_client()
        try:
            # wrap uploaded files as Parts when caching
            file_parts = _as_parts(st.session_state.pdf_files)
            if not file_parts:
                st.error("No valid file parts to cache (missing URIs).")
            else:
                cache = client.caches.create(
                    model=model,
                    contents=[types.Content(parts=file_parts)],
                    config=types.CreateCachedContentConfig(ttl_seconds=3600),  # 1 hour
                )
                st.session_state.pdf_cache_name = cache.name
                st.success("Cache created. Your next questions will reference it automatically.")
        except Exception as e:
            st.error(f"Cache creation failed: {e}")

# ---------- Input ----------
prompt = st.chat_input("Ask about your data or PDFs… (e.g., 'Summarize sections 2–3 & compare with the Excel KPIs')")

# ---------- Send ----------
if prompt:
    if not api_key:
        st.error("Paste your GEMINI_API_KEY in the sidebar.")
    else:
        add_and_render("user", prompt)

        with st.chat_message("assistant"):
            box = st.empty()
            try:
                client = ensure_client()

                cfg = types.GenerateContentConfig(temperature=temperature)
                if "2.5" in model and save_tokens:
                    cfg.thinking_config = types.ThinkingConfig(thinking_budget=0)

                # assemble and stream
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=to_gemini_messages(prompt),
                    config=cfg,
                )
                out = ""
                for chunk in stream:
                    if getattr(chunk, "text", None):
                        out += chunk.text
                        box.markdown(out)

                # Save once (avoid duplicates)
                st.session_state.history.append(("assistant", out))

            except Exception as e:
                st.error(f"⚠️ {type(e).__name__}: {e}")
