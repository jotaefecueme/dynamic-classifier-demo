FROM python:3.11-slim

WORKDIR /app

COPY demo_app.py ./
COPY requirements.txt ./
COPY .env ./

RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "demo_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
