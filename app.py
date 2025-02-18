import streamlit as st
import requests
import base64
import json
import io
from gtts import gTTS
from duckduckgo_search import DDGS
import speech_recognition as sr


# --- Constants ---
OLLAMA_HOST = "http://192.168.0.119:11434"  # Replace with your Ollama server address
OLLAMA_MODEL = "llama3.2-vision"
SYSTEM_PROMPT = """
You are an expert analyst. Analyze the given information and question.
Provide a response in JSON format with the following keys:

- "probability": A percentage (0-100) indicating the likelihood of "yes" (should do/buy) vs. "no" (should not do/buy).
                   Higher percentage means "yes", lower percentage means "no".
- "reason": A concise explanation of your analysis and reasoning, citing sources if applicable.

Example:
{"probability": 75, "reason": "Based on the current market trends and expert opinions, the stock shows strong potential for growth."}
"""


# --- Helper Functions ---

def text_to_speech(text, language="en"):
    """Converts text to speech using gTTS and returns bytes."""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"gTTS Error: {e}")
        return None

def encode_image(image_bytes):
    """Encodes image bytes to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")

def perform_ddg_search(query, max_results=3):
    """Performs DuckDuckGo search and returns concatenated results."""
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=max_results)]
        return "\n\n".join(results)
    except Exception as e:
        st.error(f"DuckDuckGo Search Error: {e}")
        return ""

def call_ollama_api(messages, stream=False, format="json"):
    """Calls the Ollama API with the given messages."""
    data = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "format": format,
        "options": {
            "temperature": st.session_state.temperature,
            "num_predict": st.session_state.max_tokens,
        }
    }
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API Error: {e}")
        return None

def transcribe_audio(language_code):
    """Transcribes audio from the microphone using speech_recognition."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("Listening...", icon="🎙️")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
        st.toast("Processing...", icon="⏳")
        try:
            text = r.recognize_google(audio, language=language_code)
            return text
        except sr.UnknownValueError:
            st.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition; {e}")
            return None


# --- Analysis Functions ---

def analyze_image(image_data, question, language):
    """Analyzes image and question using Ollama and returns probability, reason, and audio."""
    search_results = perform_ddg_search(question, max_results=2)
    combined_input = f"{question}\n\nRelevant information:\n{search_results}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input, "images": [image_data]},
    ]
    response_json = call_ollama_api(messages)
    if response_json:
        return process_ollama_response(response_json, language)
    else:
        return None, "Error: Ollama API call failed.", None

def analyze_text(question, language):
    """Analyzes text question using Ollama and returns probability, reason, and audio."""
    search_results = perform_ddg_search(question)
    combined_input = f"{question}\n\nRelevant information:\n{search_results}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input},
    ]
    response_json = call_ollama_api(messages)
    if response_json:
        return process_ollama_response(response_json, language)
    else:
        return None, "Error: Ollama API call failed.", None


def process_ollama_response(response_json, language):
    """Processes the Ollama API response, extracts data, and generates TTS."""
    try:
        content_str = response_json['message']['content']
        content_json = json.loads(content_str)
        probability = content_json.get("probability", None)
        reason = content_json.get("reason", None)

        if probability is not None and not (0 <= probability <= 100):
            raise ValueError("Probability is not within the range 0-100")

        audio_bytes = text_to_speech(reason, language) if reason else None

        return probability, reason, audio_bytes
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        st.error(f"Error processing Ollama response: {e}")
        return None, "Error: Invalid response from Ollama.", None


# --- Streamlit App ---

st.set_page_config(page_title="Should I...?", page_icon="🤔", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
/* (Your CSS styles here -  from the previous response,  unchanged) */
/* Main container */
.main {
    padding: 2rem;
    font-family: sans-serif; /* Modern font */
}

/* Title */
.title {
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #333;  /* Darker title color */
}

/* Subtitle/Description */
.subtitle {
    font-size: 1.25rem;
    color: #555;
    margin-bottom: 2rem;
}

/* Input area */
.stTextInput, .stFileUploader, .stCameraInput, .stRadio {
    margin-bottom: 1rem;
}

/* Buttons */
.stButton>button {
    background-color: #007bff; /* Primary blue color */
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem; /* Rounded corners */
    font-weight: bold;
    transition: background-color 0.2s ease; /* Smooth hover effect */
}
.stButton>button:hover {
    background-color: #0056b3; /* Darker blue on hover */
}
.stButton>button:focus {
    outline: none;  /* Remove the focus outline */
    box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25); /* Add a subtle focus ring */
}

/* Radio buttons */
.stRadio > div > label { /* Target radio button labels */
  margin-right: 1rem; /* Add spacing */
  padding: 0.5rem 1rem;
  border-radius: 0.5rem; /* Rounded */
  border: 1px solid #ccc;
  transition: all 0.2s ease;  /* Smooth transition */
}
/* Style the selected radio option */
.stRadio > div > label[data-baseweb="radio"] > div:first-child {
    background-color: #007bff !important;  /* Override to ensure color */
    border-color: #007bff !important;
}
.stRadio > div > label[data-baseweb="radio"] {
    background-color: #f0f2f6;
    border-color: #f0f2f6;
}


/* Result area */
.result-area {
    padding: 1.5rem;
    border-radius: 0.75rem;
    background-color: #f8f9fa; /* Light background */
    border: 1px solid #e9ecef;  /* Subtle border */
}

/* Success/Error messages */
.st-success, .st-error {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    font-weight: bold;
    color: white;
    display: flex;
    align-items: center;
}
.st-success {
    background-color: #28a745; /* Green */
}
.st-error {
    background-color: #dc3545; /* Red */
}
.st-success svg, .st-error svg { /* Icons within success/error */
    margin-right: 0.5rem;  /* Space the icon */
    font-size: 1.2em;
}


/* Expander */
.st-expander {
  border: none !important; /* No borders */
}
.st-expanderHeader {
    font-weight: bold;
    background-color: transparent;
    padding: 0.5rem 0;
}
.st-expanderContent {
    padding: 0.5rem 0;
}

/* Gauge/Progress */
.stProgress > div > div > div > div {
    background-color: #007bff; /* Consistent blue */
}

/* Sidebar */
.sidebar .sidebar-content {
    padding: 1rem;
    background-color: #f0f2f6; /* Lighter background */
}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "language" not in st.session_state:
    st.session_state["language"] = "en"
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 256
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.language = st.selectbox("Language", ["en", "ko", "ja", "zh-CN", "fr", "de"], index=0)
    language_code = st.session_state.language.split("-")[0]

    with st.expander("LLM Settings"):
        st.session_state.max_tokens = st.slider("Max Tokens", 1, 2048, 256, 1)
        st.session_state.temperature = st.slider("Temperature", 0.1, 4.0, 0.7, 0.1)

# --- Main App ---
st.markdown("<h1 class='title'>Should I...? 🤔</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask a question and get a probability-based analysis!</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    input_type = st.radio("Input Type", ["Text", "Voice", "Upload Image", "Take Photo"], horizontal=True)
    question = ""  # Initialize question outside the conditional blocks

    if input_type == "Text":
        question = st.text_input("Ask a question:", placeholder="e.g., Should I buy this stock?")

    elif input_type == "Voice":
        if st.button("🎤 Record Question"):
            question = transcribe_audio(language_code)
            if question:
                st.write("You said:", question)

    uploaded_image = None
    camera_image = None

    if input_type == "Upload Image":
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility='collapsed')

    elif input_type == "Take Photo":
        camera_image = st.camera_input("Take Photo", label_visibility="collapsed")

    if st.button("Analyze", type="primary", use_container_width=True):
        if not question and input_type in ("Text", "Voice") and not uploaded_image and not camera_image:
            st.warning("Please enter a question, record audio, or upload/take an image.")
        else: # No need to use continue. Use else.
            with st.spinner("Analyzing..."):
                if input_type in ("Text", "Voice") and question:
                    probability, reason, audio = analyze_text(question, st.session_state.language)

                elif input_type == "Upload Image" and uploaded_image:
                    image_bytes = uploaded_image.getvalue()
                    image_base64 = encode_image(image_bytes)
                    probability, reason, audio = analyze_image(image_base64, question, st.session_state.language)

                elif input_type == "Take Photo" and camera_image:
                    image_bytes = camera_image.getvalue()
                    image_base64 = encode_image(image_bytes)
                    probability, reason, audio = analyze_image(image_base64, question, st.session_state.language)

                else:
                    probability, reason, audio = None, "No input provided.", None  # Handle no input case

                with col2:
                    if probability is not None and reason:
                        st.subheader("Analysis Result")
                        with st.container(border=True):
                            if probability >= 50:
                                st.success(f"✅ Yes! ({probability}%)")
                                #st.balloons() # Moved outside, so no conflict.

                            else:
                                st.error(f"❌ No! ({probability}%)")
                            st.progress(probability / 100.0)
                            with st.expander("Reason", expanded=True):
                                st.markdown(reason)
                            if audio:
                                st.audio(audio, format="audio/mp3")
                    elif reason:  # Display error message
                        st.error(reason)