FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей для сборки, если потребуются
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Экспонируем порты для API и Streamlit
EXPOSE 8000
EXPOSE 8501

# По умолчанию запускаем FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
