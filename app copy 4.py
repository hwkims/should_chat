import streamlit as st
import requests
import base64
import json
import io
from gtts import gTTS
from duckduckgo_search import DDGS  # DuckDuckGo 검색

# --- Constants ---
OLLAMA_HOST = "http://192.168.0.119:11434"
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
        print(f"gTTS Error: {e}")
        return None

def encode_image(image_bytes):
    """Encodes image bytes to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")

def perform_ddg_search(query, max_results=3):
    """Performs DuckDuckGo search and returns concatenated results."""
    with DDGS() as ddgs:
        results = [r["body"] for r in ddgs.text(query, max_results=max_results)]
    return "\n\n".join(results)

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
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=data)
    response.raise_for_status()
    return response.json()


# --- Analysis Functions ---
def analyze_image(image_data, question, language):
    """Analyzes image and question using Ollama and returns probability, reason, and audio."""
    search_results = perform_ddg_search(question, max_results=2) # 검색 결과 추가.
    combined_input = f"{question}\n\nRelevant information:\n{search_results}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input, "images": [image_data]}, # 질문 + 검색결과
    ]
    response_json = call_ollama_api(messages)
    return process_ollama_response(response_json, language)

def analyze_text(question, language):
    """Analyzes text question using Ollama (with DDG search) and returns probability, reason, and audio."""
    search_results = perform_ddg_search(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\nRelevant information:\n{search_results}"},
    ]
    response_json = call_ollama_api(messages)
    return process_ollama_response(response_json, language)

def process_ollama_response(response_json, language):
    """Processes the Ollama API response, extracts data, and generates TTS."""
    try:
        content_str = response_json['message']['content']
        content_json = json.loads(content_str)
        probability = content_json.get("probability", None)
        reason = content_json.get("reason", None)

        audio_bytes = text_to_speech(reason, language) if reason else None  # type: ignore

        return probability, reason, audio_bytes
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing Ollama response: {e}")
        return None, "Error: Invalid response from Ollama.", None

# --- Streamlit App ---
# Initialize session state
if "language" not in st.session_state:
    st.session_state["language"] = "en"
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 256
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7


st.set_page_config(page_title="Should I...?", page_icon="🤔", layout="wide")
st.title("Should I...? 🤔")
st.write("Ask a question and get a probability-based analysis!")

# Sidebar (Settings)
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.language = st.selectbox("Language", ["en", "ko", "ja", "zh-CN", "fr", "de"], index=0)

    with st.expander("LLM Settings"):
        st.session_state.max_tokens = st.slider("Max Tokens", 1, 2048, 256, 1, help="Maximum number of tokens to generate.")
        st.session_state.temperature = st.slider("Temperature", 0.1, 4.0, 0.7, 0.1, help="Controls the randomness of the generation.")


# Main area layout
col1, col2 = st.columns([3, 1])  # Input and settings, Result

with col1:
    # Input Section
    question = st.text_input("Ask a question:", placeholder="e.g., Should I buy this stock? Should I invest in Bitcoin?")

    # Radio buttons for input type selection
    input_type = st.radio("Input Type", ["Text Only", "Upload Image", "Take Photo"], horizontal=True)

    if input_type == "Upload Image":
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility='collapsed')
        camera_image = None # 카메라 이미지는 일단 None
    elif input_type == "Take Photo":
        camera_image = st.camera_input("Take Photo", label_visibility="collapsed")
        uploaded_image = None # 업로드 이미지는 None
    else: # Text Only
        uploaded_image = None
        camera_image = None

    if st.button("Analyze", type="primary", use_container_width=True):
        if question:
            with st.spinner("Analyzing..."):
                if input_type == "Text Only":
                    probability, reason, audio = analyze_text(question, st.session_state.language)
                elif uploaded_image:
                    image_bytes = uploaded_image.getvalue()
                    image_base64 = encode_image(image_bytes)
                    probability, reason, audio = analyze_image(image_base64, question, st.session_state.language)
                elif camera_image:
                    image_bytes = camera_image.getvalue()
                    image_base64 = encode_image(image_bytes)
                    probability, reason, audio = analyze_image(image_base64, question, st.session_state.language)
                else:
                    probability, reason, audio = None, None, None

            if probability is not None and reason:
                # Display Results
                with col2:
                    st.subheader("Analysis Result")
                    if probability >= 50:
                        st.success(f"✅ Yes! ({probability}%)")  # Success (green)
                        st.balloons()
                    else:
                        st.error(f"❌ No! ({probability}%)")  # Error (red)

                    # 게이지를 success, error 박스 안에 넣기
                    # st.progress(probability / 100.0) # progress bar는 success, error와 함께 사용 불가.

                    # Reason
                    with st.expander("Reason", expanded=True):
                        st.markdown(reason)

                    # Audio
                    if audio:
                        st.audio(audio, format="audio/mp3")
            elif reason:
                st.error(reason)  # Display error message

        else: # 질문이 없을 때
            st.warning("Please enter a question.")