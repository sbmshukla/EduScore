# FROM python:3.9-slim
# WORKDIR /app
# COPY . /app

# RUN apt update -y && apt install awscli -y

# RUN pip install -r requirements.txt
# CMD ["python", "application.py"]


FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up APT cache
RUN apt-get update && apt-get install -y awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies without caching
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["python", "application.py"]

