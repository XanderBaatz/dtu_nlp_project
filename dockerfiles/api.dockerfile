FROM python:3.12-alpine

WORKDIR /app

COPY src/web_sentiment_analysis/main.py /app/main.py
COPY dockerfiles/sentiment_api_requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
