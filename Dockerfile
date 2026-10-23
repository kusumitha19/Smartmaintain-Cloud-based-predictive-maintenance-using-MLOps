FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN apt-get update && pip cache purge && pip install --no-cache-dir --default-timeout=1000 -r requirements-inference.txt
EXPOSE 8080

CMD ["python3", "inference_app.py"]
