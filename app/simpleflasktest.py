from flask import Flask, request, jsonify
from vllm.config import DeviceConfig
from langchain_community.llms import VLLM
import os
from huggingface_hub import login  # Hugging Face Hub에 로그인하기 위한 라이브러리
import torch


# Flask 애플리케이션 초기화
app = Flask(__name__)

# 환경 변수에서 Hugging Face API 토큰 및 모델 ID 가져오기
token = os.getenv("HF_TOKEN")
model_id = os.getenv("MODEL_ID", "explodinggradients/Ragas-critic-llm-Qwen1.5-GPTQ")  # 환경 변수에서 모델 ID를 가져오며 기본값 설정

# Hugging Face API 토큰을 이용한 로그인
if token:
    login(token)  # 토큰을 사용해 Hugging Face Hub에 로그인


# CPU가 있으면 CPU, 없으면 GPU로 설정
if torch.cuda.is_available():
    device_config = DeviceConfig(device="cuda")  # GPU 설정
else:
    device_config = DeviceConfig(device="cpu")  # CPU 설정


# VLLM 인스턴스 생성
llm = VLLM(
    model=model_id,
    trust_remote_code=True,  # Hugging Face 모델을 사용할 때 필수 옵션
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    temperature=0.0,
    device_config=device_config 
)

# 템플릿 구성
template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

@app.route('/generate', methods=['POST'])
def generate():
    # JSON에서 입력 텍스트 가져오기
    data = request.get_json()
    input_text = data.get('inputs', '')

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    # 입력 텍스트를 템플릿에 삽입
    simple_instruction = template.format(input_text)

    # VLLM을 사용하여 응답 생성
    response = llm.invoke(simple_instruction)
    
    # 응답을 JSON 형식으로 반환
    return jsonify({"response": response})

if __name__ == '__main__':
    # Flask 애플리케이션 실행
    app.run(host='0.0.0.0', port=8000, use_reloader=False)
