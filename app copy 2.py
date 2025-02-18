import streamlit as st
import requests
import base64
import json
import io
from gtts import gTTS  # gTTS 사용

# Ollama API 설정
OLLAMA_HOST = "http://192.168.0.119:11434"
OLLAMA_MODEL = "llama3.2-vision"

# 시스템 프롬프트
SYSTEM_PROMPT = """
You are an expert analyst. Analyze the given image and question.
Provide a response in JSON format with the following keys:

- "probability": A percentage (0-100) indicating the likelihood of "yes" (should do/buy) vs. "no" (should not do/buy).
                   Higher percentage means "yes", lower percentage means "no".
- "reason": A concise explanation of your analysis and reasoning.

Example:
{"probability": 75, "reason": "The stock chart shows an upward trend with increasing volume, indicating a good buying opportunity."}
"""

def text_to_speech(text, language="en"):
    """gTTS를 사용하여 텍스트를 음성으로 변환하고 bytes 형태로 반환."""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)  # 버퍼 시작 위치로
        return buffer.read()
    except Exception as e:
        print(f"gTTS Error: {e}")
        return None

def analyze_image(image_data, question, language, max_tokens, temperature, top_p):
    """Ollama API를 호출하여 이미지, 질문 분석, JSON 응답 반환."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question, "images": [image_data]},
    ]

    data = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=data)
        response.raise_for_status()
        response_json = response.json()
        content_str = response_json['message']['content']

        try:
            content_json = json.loads(content_str)
            probability = content_json.get("probability", None)
            reason = content_json.get("reason", None)

            # gTTS TTS
            audio_output = None
            audio_bytes = text_to_speech(reason, language)  # type: ignore
            if audio_bytes:
                audio_output = audio_bytes

            return probability, reason, audio_output
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None, "Error: Invalid JSON response from Ollama.", None

    except requests.exceptions.RequestException as e:
        print(f"Ollama API request failed: {e}")
        return None, "Error: Could not communicate with Ollama API.", None


# Streamlit 앱 설정
st.set_page_config(page_title="Should I...?", page_icon="🤔")
st.title("Should I...? 🤔")
st.write("Upload an image and ask a question to get a probability-based analysis!")

# 사이드바 (설정)
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Language", ["en", "ko", "ja", "zh-CN", "fr", "de"], index=0) # 수정: gTTS 지원 언어

    st.header("LLM Settings")
    max_tokens = st.slider("Max Tokens", 1, 2048, 256, 1)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)  # Ollama에서 지원하는 파라미터로

# 이미지 업로드
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# 질문 입력
question = st.text_input("Ask a question (e.g., Should I buy this stock? Should I buy this product?)")

if uploaded_image and question:
    image_bytes = uploaded_image.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    probability, reason, audio = analyze_image(image_base64, question, language, max_tokens, temperature, top_p)

    if probability is not None and reason: #type: ignore
        if probability >= 50:
            color = "green"
            emoji = "✅"
        else:
            color = "red"
            emoji = "❌"

        st.progress(probability / 100.0, text=f"{emoji} Probability: {probability}%")
        st.markdown(f"**Reason:** {reason}")

        if audio:
            st.audio(audio, format="audio/mp3")  # 수정: gTTS는 mp3

    else:
        st.error(reason)

elif uploaded_image or question:
    st.warning("Please enter *both* an image and a question.")