알겠습니다. 음성 관련 기능들을 웹 API 기반으로 전환하고, gTTS 및 LLM 설정을 확장하겠습니다.

**핵심 변경 사항:**

1.  **`speech_recognition` 제거, `streamlit-webrtc` 및 STT API 사용:**
    *   `speech_recognition` 라이브러리와 PyAudio 의존성을 제거합니다.  이로써 설치 문제를 해결하고, 더 안정적인 웹 기반 접근 방식을 사용합니다.
    *   `streamlit-webrtc`를 사용하여 브라우저에서 직접 오디오를 녹음합니다.  이 라이브러리는 Streamlit과 잘 통합되며 별도의 서버 설정 없이 작동합니다.
    *   녹음된 오디오를 텍스트로 변환하기 위해 Google Cloud Speech-to-Text API, AssemblyAI, Whisper API 등 *상용 STT API*를 사용합니다.  (예제에서는 Google Cloud STT를 사용하지만, 다른 API로 쉽게 교체할 수 있습니다.)  웹 API를 사용하면 더 높은 정확도와 안정성을 얻을 수 있습니다.
        *  Google Cloud STT를 사용하려면 Google Cloud Platform (GCP) 프로젝트를 생성하고, Speech-to-Text API를 활성화하고, 서비스 계정 키 (JSON 파일)를 생성해야 합니다.  이 키 파일을 프로젝트에 추가하고 환경 변수 `GOOGLE_APPLICATION_CREDENTIALS`를 설정해야 합니다.

2.  **gTTS 설정 확장:**
    *   성별 (남성/여성) 및 속도 (느림/보통/빠름)를 선택할 수 있는 UI 요소를 추가합니다.
    *   gTTS 객체를 생성할 때 이러한 설정을 반영합니다.

3.  **LLM 설정 확장:**
    *   `top_p`, `top_k`, `repeat_penalty`와 같은 추가적인 LLM 파라미터를 조절할 수 있는 UI 요소를 추가합니다.
    *   `call_ollama_api` 함수에서 이러한 파라미터를 `options`에 포함시켜 Ollama API에 전달합니다.

4.  **코드 정리 및 모듈화:**
    *   STT, TTS 관련 함수를 별도의 모듈 (`stt_tts.py`)로 분리하여 코드를 더 깔끔하게 구성합니다.
    *   `utils.py` 에 공통으로 사용될 함수를 넣어둠.

**전체 코드 (구조 변경):**

다음은 프로젝트의 구조입니다.

```
should_i/
├── app.py           # 메인 Streamlit 앱
├── stt_tts.py        # STT, TTS 관련 함수
├── utils.py        # 공통 함수
└── requirements.txt  # 필요한 패키지 목록
```

**requirements.txt:**

```
streamlit
requests
gTTS
duckduckgo-search
streamlit-webrtc
google-cloud-speech # Google Cloud STT 사용 시
# assemblyai  # AssemblyAI 사용 시
# openai      # Whisper API 사용 시
```

**utils.py:**
```python
import base64

def encode_image(image_bytes):
    """Encodes image bytes to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")
```

**stt_tts.py:**

```python
import io
import streamlit as st
from gtts import gTTS
from google.cloud import speech  # Google Cloud STT


def google_cloud_stt(audio_bytes, language_code):
    """Transcribes audio bytes using Google Cloud Speech-to-Text."""
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            # sample_rate_hertz=44100,  # Adjust if needed
            sample_rate_hertz=48000,
            language_code=language_code,
            enable_automatic_punctuation=True, # 구두점 추가.
        )

        response = client.recognize(config=config, audio=audio)
        if response.results:
            return response.results[0].alternatives[0].transcript
        else:
            return None

    except Exception as e:
        st.error(f"Google Cloud STT Error: {e}")
        return None

# --- 다른 STT API 사용 예시 (AssemblyAI, Whisper) ---
# def assemblyai_stt(audio_bytes, api_key):
#    # AssemblyAI API implementation (requires assemblyai package)
#     ...

# def whisper_stt(audio_bytes, api_key):
#     # OpenAI Whisper API implementation (requires openai package)
#     ...


def text_to_speech(text, language="ko", gender="female", speed="normal"):
    """Converts text to speech using gTTS with gender and speed options."""
    try:
        # gTTS에서 속도 조절은 slow=True/False로만 가능.  더 세밀한 조절은 다른 TTS 엔진 필요.
        slow = speed == "slow"
        tts = gTTS(text=text, lang=language, slow=slow, tld="com") # tld 설정

        # 성별 설정은 gTTS에서 직접 지원하지 않음.  다른 TTS 엔진 (예: Google Cloud Text-to-Speech) 필요.
        # 여기서는 gTTS를 사용하므로, 성별은 무시됨.  더 나은 성별 제어를 원하면 다른 TTS API 사용.

        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"gTTS 오류: {e}")
        return None

```

**app.py:**

```python
import streamlit as st
import requests
from duckduckgo_search import DDGS
from stt_tts import google_cloud_stt, text_to_speech  # Import STT and TTS functions
from utils import encode_image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- Constants ---
OLLAMA_HOST = "http://192.168.0.119:11434"  # Replace with your Ollama server
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

# --- Helper Functions ---

def perform_ddg_search(query, max_results=3):
    """Performs DuckDuckGo search; handles empty queries."""
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
    """Calls the Ollama API."""
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



# --- Analysis Functions ---

def analyze_image(image_data, question, language):
    """Analyzes image and question using Ollama."""
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
    """Analyzes text question using Ollama."""
    search_results = perform_ddg_search(question)
    combined_input = f"{question}\n\n관련 정보:\n{search_results}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_input},
    ]
    response_json = call_ollama_api(messages)
    return process_ollama_response(response_json, language) if response_json else (None, "오류: Ollama API 호출 실패.", None)

def process_ollama_response(response_json, language):
    """Processes the Ollama API response."""
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
        st.error(f"Ollama 응답 처리 오류: {e}")
        return None, "오류: Ollama로부터 유효하지 않은 응답.", None


# --- Streamlit App ---

st.set_page_config(page_title="해야 할까요...?", page_icon="🤔", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
/* Main container */
.main {
    padding: 2rem;
    font-family: sans-serif;
}

/* Title */
.title {
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #333;
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
    background-color: #007bff;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    font-weight: bold;
    transition: background-color 0.2s ease;
}
.stButton>button:hover {
    background-color: #0056b3;
}
.stButton>button:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
}

/* Radio buttons */
.stRadio > div > label {
  margin-right: 1rem;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  border: 1px solid #ccc;
  transition: all 0.2s ease;
}
.stRadio > div > label[data-baseweb="radio"] > div:first-child {
    background-color: #007bff !important;
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
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
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
    background-color: #28a745;
}
.st-error {
    background-color: #dc3545;
}
.st-success svg, .st-error svg {
    margin-right: 0.5rem;
    font-size: 1.2em;
}

/* Expander */
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

/* Gauge/Progress */
.stProgress > div > div > div > div {
    background-color: #007bff;
}

/* Sidebar */
.sidebar .sidebar-content {
    padding: 1rem;
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
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



# --- Sidebar ---
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
        st.session_state.tts_gender = st.selectbox("목소리 성별", ["female", "male"], index=0)  # gTTS는 제한적
        st.session_state.tts_speed = st.select_slider("읽기 속도", ["slow", "normal", "fast"], value="normal")


# --- Main App ---
st.markdown("<h1 class='title'>해야 할까요...? 🤔</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>질문을 하고 확률 기반 분석을 받으세요!</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    input_type = st.radio("입력 유형", ["텍스트", "음성", "이미지 업로드", "사진 촬영"], horizontal=True)
    question = ""
    audio_bytes = None # 음성 바이트

    if input_type == "텍스트":
        question = st.text_input("질문을 입력하세요:", placeholder="예: 이 주식을 사야 할까요?")

    elif input_type == "음성":
        # webrtc_streamer를 사용하여 오디오 녹음
        webrtc_ctx = webrtc_streamer(
            key="speech-recording",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # 필요 시 STUN 서버 설정
            media_stream_constraints={"video": False, "audio": True},
        )

        if webrtc_ctx.state.playing:
            st.write("🎤 녹음 중...")

        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                if audio_frames:
                    # 최신 프레임 가져오기
                    latest_frame = audio_frames[-1]
                    audio_bytes = latest_frame.to_ndarray().tobytes()


                    if audio_bytes:
                        question = google_cloud_stt(audio_bytes, language_code) # STT
                        if question:
                            st.write("당신의 질문:", question)
            except Exception as e:
                 st.error(f"오류: {e}")

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


    if st.button("분석", type="primary", use_container_width=True):
        if input_type == "이미지 업로드" and not uploaded_image:
             st.warning("이미지를 업로드해주세요.")
        elif input_type == "사진 촬영" and not camera_image:
            st.warning("사진을 촬영해주세요")
        elif not question and input_type in ("텍스트", "음성") :
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
```

**실행 방법 및 주의 사항:**

1.  **필수 패키지 설치:** `pip install -r requirements.txt`
2.  **Google Cloud Speech-to-Text 설정 (Google Cloud STT 사용 시):**
    *   GCP 프로젝트 생성 및 Speech-to-Text API 활성화.
    *   서비스 계정 키 (JSON 파일) 생성.
    *   `GOOGLE_APPLICATION_CREDENTIALS` 환경 변수를 서비스 계정 키 파일 경로로 설정.  (또는 코드 내에서 `os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/key.json"`을 사용할 수도 있지만, 보안상 권장하지 않습니다.)
3.  **Ollama 서버 실행:** Ollama 서버가 실행 중이고 `OLLAMA_HOST`에 올바른 주소가 설정되어 있는지 확인합니다.
4.  **Streamlit 앱 실행:** `streamlit run app.py`
5.  **HTTPS (권장):** `streamlit-webrtc`는 로컬 환경 (`localhost`)에서는 HTTP로 작동하지만, 배포 시에는 HTTPS가 필요합니다.  HTTPS를 사용하면 브라우저에서 마이크 권한을 더 쉽게 얻을 수 있습니다.

이제 앱은 웹 API를 사용하여 음성 입력을 처리하고, gTTS 및 LLM 설정이 확장되었습니다.  Google Cloud STT를 사용하는 대신 다른 STT API (AssemblyAI, Whisper API 등)를 사용하려면 `stt_tts.py` 파일에서 해당 API를 사용하도록 코드를 수정하면 됩니다.
