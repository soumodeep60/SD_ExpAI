# app.py — Gemini Chat + Excel insights (Streamlit + google-genai)

import os, io, time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Experience insights app by Soumodeep", page_icon="📊", layout="centered")
st.title("Experience insights app by Soumodeep")
st.write("Powered by Google Gemini")

# ---------- Sidebar: API + model ----------
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("GEMINI_API_KEY", type="password")
model = st.sidebar.selectbox("Model", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"], 0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
save_tokens = st.sidebar.checkbox("Disable thinking (faster/cheaper)", value=True)

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

# ---------- Sidebar: How to attach data ----------
attach_mode = st.sidebar.radio(
    "Attach to prompt",
    ["Sample rows (compact)", "Stat summary (describe + groupby)"],
    index=0,
)
max_rows = st.sidebar.slider("Max rows to include (sample mode)", 10, 300, 60, 10)
group_col = st.sidebar.selectbox(
    "Group by column (summary mode)", ["<none>"] + (list(df.columns) if df is not None else []), index=0
)

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.history = []
    st.rerun()

# ---------- Chat history ----------
if "history" not in st.session_state:
    st.session_state.history = []  # list[(role, text)]

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
        # small CSV sample + schema
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
        # describe + optional groupby
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
    """Convert history + data context + new question to Gemini content dicts."""
    msgs = []
    # history
    for role, text in st.session_state.history:
        msgs.append({"role": "user" if role == "user" else "model", "parts": [{"text": text}]})
    # new question with optional data
    if user_prompt:
        if df is not None:
            data_context = build_data_context(df)
            prompt = (
                "You are a data analyst. Use the provided DATA to answer the QUESTION.\n"
                "Be concise; show key figures; if assumptions are needed, state them.\n\n"
                f"{data_context}\n\nQUESTION:\n{user_prompt}"
            )
        else:
            prompt = user_prompt
        msgs.append({"role": "user", "parts": [{"text": prompt}]})
    return msgs

# ---------- Input ----------
prompt = st.chat_input("Ask about your data… (e.g., 'Which product drives most revenue by month?')")

# ---------- Send ----------
if prompt:
    if not api_key:
        st.error("Paste your GEMINI_API_KEY in the sidebar.")
    else:
        add_and_render("user", prompt)

        with st.chat_message("assistant"):
            box = st.empty()
            try:
                os.environ["GEMINI_API_KEY"] = api_key
                client = genai.Client()

                cfg = types.GenerateContentConfig(temperature=temperature)
                if "2.5" in model and save_tokens:
                    cfg.thinking_config = types.ThinkingConfig(thinking_budget=0)

                # Retry a bit on 429s
                retries, delay = 3, 1.5
                for attempt in range(retries):
                    try:
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
                        break
                    except Exception as e:
                        if "429" in str(e) and attempt < retries - 1:
                            time.sleep(delay); delay *= 2; continue
                        raise

                # Save once (avoid duplicates)
                st.session_state.history.append(("assistant", out))

            except Exception as e:
                st.error(f"⚠️ {type(e).__name__}: {e}")
