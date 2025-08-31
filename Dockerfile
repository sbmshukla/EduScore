FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD ["python", "application.py"]
