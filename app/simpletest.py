from vllm.config import DeviceConfig
from langchain_community.llms import VLLM

# CPU로 강제 설정
device_config = DeviceConfig(device="cpu")

# VLLM 인스턴스 생성
llm = VLLM(
    model="explodinggradients/Ragas-critic-llm-Qwen1.5-GPTQ",
    trust_remote_code=True,  # Hugging Face 모델을 사용할 때 필수 옵션
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    temperature=0.0,
    device_config=device_config  # CPU 설정을 명시적으로 전달
)

# 템플릿 구성
template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
input_text = "how old are you?"

# 입력 텍스트를 템플릿에 삽입
simple_instruction = template.format(input_text)

# VLLM을 사용하여 응답 생성
response = llm.invoke(simple_instruction)
print(response)
