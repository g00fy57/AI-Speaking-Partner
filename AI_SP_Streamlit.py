import streamlit as st
from openai import OpenAI
import os
import json
from datetime import datetime
import uuid
import tempfile

# ----------------------
# Configuration / Utils
# ----------------------

APP_FILE = "ai_speaking_partner_data.json"
DEFAULT_MODEL = "gpt-4o-mini"
CORRECTION_PROMPT = (
    "You are an empathetic English teacher.\n"
    "Given the user's sentence, provide: 1) a short natural conversational reply; "
    "2) the corrected sentence (if needed); 3) a one-line grammar explanation; 4) up to 5 new vocabulary words with short definitions.\n"
    "Return the result in JSON with keys: reply, corrected, explanation, vocab (list of {word: definition}).\n"
)

client = None

# ----------------------
# Storage helpers
# ----------------------

def load_data():
    if not os.path.exists(APP_FILE):
        data = {
            "users": {},
            "challenges": [
                "Describe your last vacation.",
                "Roleplay a job interview.",
                "Explain your daily routine.",
                "Tell a short story.",
                "Describe your favourite movie.",
            ],
        }
        save_data(data)
        return data
    else:
        with open(APP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

def save_data(data):
    with open(APP_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_user_id():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

# ----------------------
# OpenAI helpers (v1.0+ syntax)
# ----------------------

def ensure_api_key(api_key_input):
    api_key = api_key_input or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key is required.")
        st.stop()
    global client
    client = OpenAI(api_key=api_key)
    return client


def call_chat_model(system_prompt, user_prompt, model=DEFAULT_MODEL):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""


def call_correction_and_vocab(user_sentence, model=DEFAULT_MODEL):
    prompt = CORRECTION_PROMPT + f"\nUser sentence: \"{user_sentence}\"\nReturn only JSON."
    raw = call_chat_model("You are a careful English teacher.", prompt, model=model)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        json_text = raw[start:end+1] if start != -1 and end != -1 else raw
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        return {"reply": raw, "corrected": user_sentence, "explanation": "Parsing error", "vocab": []}


def generate_reply_for_conversation(history, user_message, model=DEFAULT_MODEL):
    messages = [{"role": "system", "content": "You are an empathetic conversational partner for English learners."}]
    for role, text in history:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_message})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return ""


def transcribe_audio_file(audio_file_path):
    try:
        with open(audio_file_path, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            return resp.text
    except Exception as e:
        st.error(f"Audio transcription error: {e}")
        return ""

# ----------------------
# App UI
# ----------------------

st.set_page_config(page_title="AI Speaking Partner", page_icon="🗣️")
st.title("🗣️ AI Speaking Partner — English Practice")

api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
client = ensure_api_key(api_key_input)

data = load_data()
user_id = get_user_id()
if user_id not in data["users"]:
    data["users"][user_id] = {"sessions": [], "vocab_known": []}

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Conversation")
for role, text in st.session_state.history[-20:]:
    st.markdown(f"**{role.capitalize()}:** {text}")

user_input = st.text_area("Say something", height=100)
if st.button("Send"):
    if user_input.strip():
        conv_reply = generate_reply_for_conversation(st.session_state.history, user_input)
        st.session_state.history.append(("user", user_input))
        corr = call_correction_and_vocab(user_input)
        assistant_text = f"{conv_reply}\n\n**Correction:** {corr.get('corrected','')}\n**Why:** {corr.get('explanation','')}"
        st.session_state.history.append(("assistant", assistant_text))
        data["users"][user_id]["sessions"].append({"user": user_input, "assistant": assistant_text})
        save_data(data)
        st.rerun()

st.sidebar.write(f"Sessions: {len(data['users'][user_id]['sessions'])}")
