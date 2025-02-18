import streamlit as st
import requests
import base64
import json
import io
from gtts import gTTS
from duckduckgo_search import DDGS

# --- 상수 ---
OLLAMA_HOST = "http://192.168.0.119:11434"  # Ollama 서버 주소
OLLAMA_MODEL = "llama3.2-vision"  # 사용할 모델
SYSTEM_PROMPT = """
당신은 전문 분석가입니다. 주어진 정보와 질문을 분석하여, '예' 또는 '아니오'로 대답할 가능성을 백분율(0-100)로 제시하고, 그 이유를 설명해주세요.  해당되는 경우 출처를 인용하세요.

예시:
{"probability": 75, "reason": "현재 시장 동향과 전문가 의견에 따르면, 해당 주식은 강력한 성장 잠재력을 보입니다. 출처:구글"}
"""

# --- 도우미 함수 ---

def text_to_speech(text, language="ko", gender="female", speed="normal"):
    """gTTS를 사용하여 텍스트를 음성으로 변환 (성별, 속도 옵션)."""
    try:
        slow = speed == "slow"
        tts = gTTS(text=text, lang=language, slow=slow, tld="com")  # tld 설정

        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"gTTS 오류: {e}")
        return None

def encode_image(image_bytes):
    """이미지 바이트를 base64로 인코딩."""
    return base64.b64encode(image_bytes).decode("utf-8")

def perform_ddg_search(query, max_results=3):
    """DuckDuckGo 검색 수행 (빈 쿼리 처리)."""
    if not query:
        return ""
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=max_results)]
            return "\n\n".join(results)
    except Exception as e:
        st.error(f"DuckDuckGo 검색 오류: {e}")  # 오류 메시지 표시
        return ""

def call_ollama_api(messages, stream=False, format="json"):
    """Ollama API 호출 (LLM 옵션 포함)."""
    data = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "format": format,
        "options": {
            "temperature": st.session_state.temperature,
            "num_predict": st.session_state.max_tokens,
            "top_p": st.session_state.top_p,
            "top_k": st.session_state.top_k,
            "repeat_penalty": st.session_state.repeat_penalty,
        },
    }
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=data)
        response.raise_for_status()  # HTTP 오류 처리
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API 오류: {e}")  # 오류 메시지 표시
        return None

# --- 분석 함수 ---

def analyze_image(image_data, question, language):
    """이미지와 질문을 분석 (Ollama, 선택적 DuckDuckGo)."""
    if question:
        search_results = perform_ddg_search(question, max_results=2)
        combined_input = f"{question}\n\n관련 정보:\n{search_results}"
    else:
        combined_input = "사진에 대해 분석해주세요."  # 기본 메시지

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input, "images": [image_data]},
    ]
    response_json = call_ollama_api(messages)
    return process_ollama_response(response_json, language) if response_json else (None, "오류: Ollama API 호출 실패.", None)

def analyze_text(question, language):
    """텍스트 질문 분석 (Ollama, DuckDuckGo)."""
    search_results = perform_ddg_search(question)
    combined_input = f"{question}\n\n관련 정보:\n{search_results}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input},
    ]
    response_json = call_ollama_api(messages)
    return process_ollama_response(response_json, language) if response_json else (None, "오류: Ollama API 호출 실패.", None)

def process_ollama_response(response_json, language):
    """Ollama 응답 처리, 데이터 추출, TTS 생성."""
    try:
        content_str = response_json['message']['content']
        content_json = json.loads(content_str)
        probability = content_json.get("probability", None)
        reason = content_json.get("reason", None)

        if probability is not None and not (0 <= probability <= 100):
            raise ValueError("확률이 0-100 범위 내에 있지 않습니다.")

        audio_bytes = text_to_speech(
            reason, language, st.session_state.tts_gender, st.session_state.tts_speed
        ) if reason else None

        return probability, reason, audio_bytes
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        st.error(f"Ollama 응답 처리 오류: {e}")  # 오류 메시지 표시
        return None, "오류: Ollama로부터 유효하지 않은 응답.", None

# --- Streamlit 앱 ---

st.set_page_config(page_title="해야 할까요...?", page_icon="🤔", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
/* 전체 스타일 */
body {
    font-family: 'Noto Sans KR', sans-serif; /* 한글 폰트 */
}
.main {
    padding: 2rem;
}

/* 제목 */
.title {
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #333;
}

/* 부제목 */
.subtitle {
    font-size: 1.25rem;
    color: #555;
    margin-bottom: 2rem;
}

/* 입력 */
.stTextInput, .stFileUploader, .stRadio {
    margin-bottom: 1rem;
}

/* 버튼 */
.stButton>button {
    background-color: #4CAF50; /* 녹색 계열 */
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    font-weight: bold;
    transition: background-color 0.2s ease;
}
.stButton>button:hover {
    background-color: #367C39; /* 더 진한 녹색 */
}
.stButton>button:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.25); /* 녹색 포커스 링 */
}

/* 라디오 버튼 */
.stRadio > div > label {
    margin-right: 1rem;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ccc;
    transition: all 0.2s ease;
}
.stRadio > div > label[data-baseweb="radio"] > div:first-child {
    background-color: #4CAF50 !important; /* 선택된 라디오 버튼 */
    border-color: #4CAF50 !important;
}
.stRadio > div > label[data-baseweb="radio"] {
    background-color: #f0f2f6;
    border-color: #f0f2f6;
}

/* 결과 영역 */
.result-area {
    padding: 1.5rem;
    border-radius: 0.75rem;
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
}

/* 성공/오류 메시지 */
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
    background-color: #28a745;
}
.st-error {
    background-color: #dc3545;
}
.st-success svg, .st-error svg {
    margin-right: 0.5rem;
    font-size: 1.2em;
}

/* 확장 패널 */
.st-expander {
    border: none !important;
}
.st-expanderHeader {
    font-weight: bold;
    background-color: transparent;
    padding: 0.5rem 0;
}
.st-expanderContent {
    padding: 0.5rem 0;
}

/* 진행률 표시줄 */
.stProgress > div > div > div > div {
    background-color: #4CAF50; /* 녹색 */
}

/* 사이드바 */
.sidebar .sidebar-content {
    padding: 1rem;
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# --- 세션 상태 초기화 ---
if "language" not in st.session_state:
    st.session_state["language"] = "ko"  # 기본 언어: 한국어
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 256
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7
if "top_p" not in st.session_state:
    st.session_state["top_p"] = 0.9
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 50
if "repeat_penalty" not in st.session_state:
    st.session_state["repeat_penalty"] = 1.1
if "tts_gender" not in st.session_state:
    st.session_state["tts_gender"] = "female"
if "tts_speed" not in st.session_state:
    st.session_state["tts_speed"] = "normal"

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.session_state.language = st.selectbox("언어", ["ko", "en", "ja", "zh-CN", "fr", "de"], index=0)
    language_code = st.session_state.language.split("-")[0]  # "ko", "en" 등

    with st.expander("LLM 설정"):
        st.session_state.max_tokens = st.slider("최대 토큰 수", 1, 2048, 256, 1)
        st.session_state.temperature = st.slider("온도", 0.1, 4.0, 0.7, 0.1)
        st.session_state.top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
        st.session_state.top_k = st.slider("Top K", 1, 100, 50, 1)
        st.session_state.repeat_penalty = st.slider("반복 페널티", 0.1, 2.0, 1.1, 0.05)

    with st.expander("TTS 설정"):
        st.session_state.tts_gender = st.selectbox("목소리 성별", ["female", "male"], index=0)
        st.session_state.tts_speed = st.select_slider("읽기 속도", ["slow", "normal", "fast"], value="normal")

# --- 메인 앱 ---
st.markdown("<h1 class='title'>해야 할까요...? 🤔</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>질문을 하고 확률 기반 분석을 받으세요!</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])  # 입력 영역과 결과 영역

with col1:
    input_type = st.radio("입력 유형", ["텍스트", "이미지 업로드"], horizontal=True)
    question = ""
    uploaded_image = None

    if input_type == "텍스트":
        question = st.text_input("질문을 입력하세요:", placeholder="예: 이 주식을 사야 할까요?")
    elif input_type == "이미지 업로드":
        question = st.text_input("질문 (선택 사항):", placeholder="이미지에 대한 질문을 입력하세요 (선택 사항).")
        uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], label_visibility='collapsed')

    if st.button("분석", type="primary", use_container_width=True):
        if input_type == "이미지 업로드" and not uploaded_image:
            st.warning("이미지를 업로드해주세요.")
        elif not question and input_type == "텍스트":  # 텍스트 입력인데 질문이 없는 경우
            st.warning("질문을 입력해주세요.")
        else:
            with st.spinner("분석 중..."):
                if input_type == "텍스트" and question:
                    probability, reason, audio = analyze_text(question, st.session_state.language)
                elif input_type == "이미지 업로드" and uploaded_image:
                    image_bytes = uploaded_image.getvalue()
                    image_base64 = encode_image(image_bytes)
                    probability, reason, audio = analyze_image(image_base64, question, st.session_state.language)
                else:
                    probability, reason, audio = None, "입력이 제공되지 않았습니다.", None

                with col2:  # 결과 표시
                    if probability is not None and reason:
                        st.subheader("분석 결과")
                        with st.container(border=True):
                            if probability >= 50:
                                st.success(f"✅ 예! ({probability}%)")
                            else:
                                st.error(f"❌ 아니오! ({probability}%)")
                            st.progress(probability / 100.0)
                            with st.expander("이유", expanded=True):
                                st.markdown(reason)
                            if audio:
                                st.audio(audio, format="audio/mp3")
                    elif reason:
                        st.error(reason)