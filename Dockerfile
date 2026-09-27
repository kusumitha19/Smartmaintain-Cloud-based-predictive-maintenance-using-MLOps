FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN apt-get update && pip install --default-timeout=1000 --no-cache-dir -r requirements-inference.txt
EXPOSE 8080

CMD ["python3", "inference_app.py"]
