import streamlit as st
import requests
import base64
import json
import io
from gtts import gTTS  # gTTS는 여전히 사용 (웹 API 대안이 제한적)
from duckduckgo_search import DDGS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import logging

# 로깅 설정 (webrtc 관련 오류를 보기 위함)
logging.basicConfig(level=logging.DEBUG)

# --- 상수 ---
OLLAMA_HOST = "http://192.168.0.119:11434"  # Ollama 서버 주소
OLLAMA_MODEL = "llama3.2-vision"
SYSTEM_PROMPT = """
당신은 전문 분석가입니다. 주어진 정보와 질문을 분석하세요.
다음 키를 사용하여 JSON 형식으로 응답을 제공하세요:

- "probability": "예" (해야 함/구매해야 함) 대 "아니오" (하지 말아야 함/구매하지 말아야 함)의 가능성을 나타내는 백분율 (0-100)입니다.
                   더 높은 백분율은 "예"를 의미하고, 더 낮은 백분율은 "아니오"를 의미합니다.
- "reason": 해당되는 경우 출처를 인용하여 분석 및 추론에 대한 간결한 설명.

예시:
{"probability": 75, "reason": "현재 시장 동향과 전문가 의견에 따르면, 해당 주식은 강력한 성장 잠재력을 보입니다."}
"""

# --- 도우미 함수 ---

def text_to_speech(text, language="ko", gender="female", speed="normal"):
    """gTTS를 사용하여 텍스트를 음성으로 변환합니다 (성별, 속도 옵션 포함)."""
    try:
        slow = speed == "slow"
        tts = gTTS(text=text, lang=language, slow=slow, tld="com") # tld 설정.

        # 성별 설정은 gTTS에서 직접 지원하지 않음. 다른 TTS엔진 필요.

        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"gTTS 오류: {e}")
        return None

def encode_image(image_bytes):
    """이미지 바이트를 base64로 인코딩합니다."""
    return base64.b64encode(image_bytes).decode("utf-8")

def perform_ddg_search(query, max_results=3):
    """DuckDuckGo 검색을 수행하고 연결된 결과를 반환합니다 (빈 쿼리 처리)."""
    if not query:
        return ""
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=max_results)]
        return "\n\n".join(results)
    except Exception as e:
        st.error(f"DuckDuckGo 검색 오류: {e}")
        return ""

def call_ollama_api(messages, stream=False, format="json"):
    """Ollama API를 호출합니다 (확장된 LLM 옵션 포함)."""
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
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API 오류: {e}")
        return None

# --- 분석 함수 ---
def analyze_image(image_data, question, language):
    """이미지와 질문을 분석합니다 (Ollama 사용, 선택적 DuckDuckGo 검색)."""
    if question:
        search_results = perform_ddg_search(question, max_results=2)
        combined_input = f"{question}\n\n관련 정보:\n{search_results}"
    else:
        combined_input = "사진에 대해 분석해주세요."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input, "images": [image_data]},
    ]
    response_json = call_ollama_api(messages)
    return process_ollama_response(response_json, language) if response_json else (None, "오류: Ollama API 호출 실패.", None)

def analyze_text(question, language):
    """텍스트 질문을 분석합니다 (Ollama 사용, DuckDuckGo 검색)."""
    search_results = perform_ddg_search(question)
    combined_input = f"{question}\n\n관련 정보:\n{search_results}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input},
    ]
    response_json = call_ollama_api(messages)
    return process_ollama_response(response_json, language) if response_json else (None, "오류: Ollama API 호출 실패.", None)

def process_ollama_response(response_json, language):
    """Ollama API 응답을 처리하고, 데이터를 추출하고, TTS를 생성합니다."""
    try:
        content_str = response_json['message']['content']
        content_json = json.loads(content_str)
        probability = content_json.get("probability", None)
        reason = content_json.get("reason", None)

        if probability is not None and not (0 <= probability <= 100):
            raise ValueError("확률이 0-100 범위 내에 있지 않습니다.")

        # 선택된 성별과 속도로 TTS 생성
        audio_bytes = text_to_speech(
            reason, language, st.session_state.tts_gender, st.session_state.tts_speed
        ) if reason else None

        return probability, reason, audio_bytes
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        st.error(f"Ollama 응답 처리 오류: {e}")
        return None, "오류: Ollama로부터 유효하지 않은 응답.", None

# --- Streamlit 앱 ---

st.set_page_config(page_title="해야 할까요...?", page_icon="🤔", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
/* (이전 CSS 스타일) */
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

# --- 세션 상태 초기화 ---
if "language" not in st.session_state:
    st.session_state["language"] = "ko"
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
if "tts_gender" not in st.session_state:  # TTS 설정
    st.session_state["tts_gender"] = "female"
if "tts_speed" not in st.session_state:
    st.session_state["tts_speed"] = "normal"
if "recorded_audio" not in st.session_state:  # 녹음된 오디오 저장
    st.session_state["recorded_audio"] = None

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.session_state.language = st.selectbox("언어", ["ko", "en", "ja", "zh-CN", "fr", "de"], index=0)
    language_code = st.session_state.language.split("-")[0]

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

col1, col2 = st.columns([3, 1])

with col1:
    input_type = st.radio("입력 유형", ["텍스트", "음성", "이미지 업로드", "사진 촬영"], horizontal=True)
    question = ""

    if input_type == "텍스트":
        question = st.text_input("질문을 입력하세요:", placeholder="예: 이 주식을 사야 할까요?")

    elif input_type == "음성":
        # WebRTC를 사용하여 브라우저에서 오디오 녹음 (JavaScript 사용)
        webrtc_ctx = webrtc_streamer(
            key="speech-recording",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"video": False, "audio": True},
        )

        if webrtc_ctx.state.playing:
            st.write("🎤 녹음 중...")
        if webrtc_ctx.audio_receiver:

             # 오디오 프레임을 JavaScript에서 처리하는 것이 더 효율적 (Python 콜백 최소화)
            if webrtc_ctx.audio_receiver.frames_queue.qsize() > 0:
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames()

                    if audio_frames:
                        latest_frame = audio_frames[-1]
                        audio_bytes = latest_frame.to_ndarray().tobytes()

                        #  JavaScript로 오디오 데이터 처리
                        st.session_state.recorded_audio = audio_bytes # 세션에 저장.

                except Exception as e:
                    st.error(f"Error processing audio frames: {e}")

    if input_type == "이미지 업로드":
        question = st.text_input("질문 (선택 사항):", placeholder="이미지에 대한 질문을 입력하세요 (선택 사항).")
        uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], label_visibility='collapsed')
        camera_image = None
    elif input_type == "사진 촬영":
        question = st.text_input("질문 (선택 사항):", placeholder="사진에 대한 질문을 입력하세요 (선택 사항).")
        camera_image = st.camera_input("사진 촬영", label_visibility="collapsed")
        uploaded_image = None
    else:
        uploaded_image = None
        camera_image = None

    # JavaScript를 사용하여 Speech Recognition API 호출 (브라우저 내에서)
    if input_type == "음성" and st.session_state.recorded_audio:
        js_code = f"""
        async function transcribeAudio() {{
            const audioBlob = new Blob([new Uint8Array(JSON.parse('{json.dumps(list(st.session_state.recorded_audio))}'))], {{ type: 'audio/webm' }}); // 바이트 배열을 Blob으로
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = '{language_code}';
            recognition.continuous = false;
            recognition.interimResults = false;

            recognition.onresult = (event) => {{
                const transcript = event.results[0][0].transcript;
                console.log('Transcript:', transcript);
                // Streamlit에 텍스트 전송 (callback 사용 X - Python 변수에 직접 할당)
                document.dispatchEvent(new CustomEvent("set_question", {{detail: transcript}}));

            }};

            recognition.onerror = (event) => {{
                console.error('Recognition error:', event.error);
                 document.dispatchEvent(new CustomEvent("st_error", {{detail: event.error}}));

            }};
            recognition.onend = () => {{
                console.log('Recognition ended.');

            }}

            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }}); // Stream 불필요.
            recognition.start();


        }}
        transcribeAudio();
        """
        st.components.v1.html(f"<script>{js_code}</script>", height=0)


    # Streamlit 이벤트 리스너 (JavaScript에서 Python으로 데이터 전송)
    st.markdown(
        """
        <script>
        document.addEventListener('set_question', function(e) {
            var input = window.parent.document.querySelectorAll('input[type=text]')[0];
            input.value = e.detail;
            input.dispatchEvent(new Event('input')); // Streamlit에 input 이벤트 트리거

        });
         document.addEventListener('st_error', function(e) {
             // Streamlit python console 에 로그
            console.error('JS STT:', e.detail);

        });

        </script>
        """,
        unsafe_allow_html=True,
    )


    if st.button("분석", type="primary", use_container_width=True):
        if input_type == "이미지 업로드" and not uploaded_image:
            st.warning("이미지를 업로드해주세요.")
        elif input_type == "사진 촬영" and not camera_image:
            st.warning("사진을 촬영해주세요.")
        elif not question and input_type in ("텍스트", "음성"):
            st.warning("질문을 입력하거나, 음성을 녹음해주세요.")
        else:
            with st.spinner("분석 중..."):
                if input_type in ("텍스트", "음성") and question:
                    probability, reason, audio = analyze_text(question, st.session_state.language)
                elif input_type == "이미지 업로드" and uploaded_image:
                    image_bytes = uploaded_image.getvalue()
                    image_base64 = encode_image(image_bytes)
                    probability, reason, audio = analyze_image(image_base64, question, st.session_state.language)
                elif input_type == "사진 촬영" and camera_image:
                    image_bytes = camera_image.getvalue()
                    image_base64 = encode_image(image_bytes)
                    probability, reason, audio = analyze_image(image_base64, question, st.session_state.language)
                else:
                    probability, reason, audio = None, "입력이 제공되지 않았습니다.", None

                with col2:
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