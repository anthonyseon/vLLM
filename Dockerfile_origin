FROM python:3.11

# Rust 설치
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 환경변수 설정
ENV PATH="/root/.cargo/bin:${PATH}"

# requirements.txt 복사
COPY requirements.txt .

# pip 최신화
RUN pip install --upgrade pip

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . /app
WORKDIR /app

# 애플리케이션 실행
# CMD ["python", "app.py"]
