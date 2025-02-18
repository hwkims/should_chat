import streamlit as st
import requests
import base64
import json
import io
import torch
import torchaudio
from zonos.model import Zonos

# Zonos 모델 로드 (오류 처리)
try:
    zonos_model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device="cuda")
    zonos_model.eval()
except Exception as e:
    print(f"Error loading Zonos model: {e}")
    zonos_model = None

# Ollama API 설정
OLLAMA_HOST = "http://192.168.0.119:11434"  # Ollama 서버 주소
OLLAMA_MODEL = "llama3.2-vision"  # 사용할 모델


# 시스템 프롬프트
SYSTEM_PROMPT = """
You are an expert analyst. Analyze the given image and question.
Provide a response in JSON format with the following keys:

- "probability":  A percentage (0-100) indicating the likelihood of "yes" (should do/buy) vs. "no" (should not do/buy).
                    Higher percentage means "yes", lower percentage means "no".
- "reason": A concise explanation of your analysis and reasoning.

Example:
{"probability": 75, "reason": "The stock chart shows an upward trend with increasing volume, indicating a good buying opportunity."}
"""
def text_to_speech(text, language="en-us", speed=1.0, pitch_variation=0.5, max_frequency=44000, audio_quality=0.7, emotion="neutral"):
    """Zonos TTS를 사용하여 텍스트를 음성으로 변환하고, bytes 형태로 반환."""
    if zonos_model is None:
        return None

    with torch.no_grad():
        try:
            dummy_wav = torch.randn(1, 16000)
            dummy_sr = 16000
            dummy_speaker = zonos_model.make_speaker_embedding(dummy_wav, dummy_sr)
            cond_dict = make_cond_dict(
                text=text,
                speaker=dummy_speaker,
                language=language,
                speed=speed,
                pitch_variation = pitch_variation,
                max_frequency=max_frequency,
                audio_quality=audio_quality,
                emotion=emotion
            )
            conditioning = zonos_model.prepare_conditioning(cond_dict)
            codes = zonos_model.generate(conditioning)
            wavs = zonos_model.autoencoder.decode(codes).cpu()

            # BytesIO를 사용하여 오디오 데이터를 bytes 형태로 변환
            buffer = io.BytesIO()
            torchaudio.save(buffer, wavs[0], zonos_model.autoencoder.sampling_rate, format="wav")
            return buffer.getvalue()

        except Exception as e:
            print(f"TTS Error: {e}")
            return None


def analyze_image(image_data, question, language, speed, pitch_variation, max_frequency, audio_quality, emotion, max_tokens, temperature, top_p):
    """Ollama API를 호출하여 이미지와 질문을 분석하고, JSON 응답을 반환합니다."""
    # print(f"analyze_image : {question}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question, "images": [image_data]},
    ]

    data = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,  # JSON 응답 전체를 받기 위해 False
        "format": "json",  # JSON 형식 지정
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=data)
        response.raise_for_status()
        response_json = response.json()
        # print("response_json",response_json) # 디버깅

        content_str = response_json['message']['content']
        # print("content_str:", content_str)

        # JSON 문자열 파싱
        try:
            content_json = json.loads(content_str)
            probability = content_json.get("probability", None)
            reason = content_json.get("reason", None)

            #Zonos TTS
            audio_output = None
            audio_bytes = text_to_speech(reason, language,speed, pitch_variation,max_frequency,audio_quality, emotion) # type: ignore
            if audio_bytes:
                audio_output = audio_bytes  # 오디오 데이터 저장

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

#사이드바
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Language", ["en-us", "ja-jp", "zh-cn", "fr-fr", "de-de"], index=0)
    speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
    pitch_variation = st.slider("Pitch Variation", 0.0, 1.0, 0.5, 0.1)
    max_frequency = st.slider("Max Frequency", 5000, 44000, 44000, 1000)
    audio_quality = st.slider("Audio Quality", 0.0, 1.0, 0.7, 0.05)
    emotion = st.selectbox("Emotion", ["neutral", "happy", "sad", "angry", "fearful"], index=0)

    st.header("LLM Settings")
    max_tokens = st.slider("Max Tokens", 1, 2048, 256, 1)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)



# 이미지 업로드
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# 질문 입력
question = st.text_input("Ask a question (e.g., Should I buy this stock? Should I buy this product?)")

if uploaded_image and question:
    # 이미지를 base64로 인코딩
    image_bytes = uploaded_image.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Ollama API 호출
    probability, reason, audio = analyze_image(image_base64, question, language, speed, pitch_variation, max_frequency, audio_quality, emotion, max_tokens, temperature, top_p)


    if probability is not None and reason: # type: ignore
        # 메트로놈 스타일 게이지 표시
        if probability >= 50:
            color = "green"
            emoji = "✅"  # 긍정 이모지
        else:
            color = "red"
            emoji = "❌"  # 부정 이모지

        st.progress(probability / 100.0, text=f"{emoji} Probability: {probability}%")

        # 분석 이유 표시
        st.markdown(f"**Reason:** {reason}")

        if audio:
            st.audio(audio, format="audio/wav")


    else:
        st.error(reason)  # 오류 메시지 표시

elif uploaded_image or question:
    # 이미지만 업로드하거나, 질문만 입력했을때
    st.warning("이미지와 질문을 *모두* 입력해주세요(Please enter *both* an image and a question.)")