# FROM python:3.9-slim
# WORKDIR /app
# COPY . /app

# RUN apt update -y && apt install awscli -y

# RUN pip install -r requirements.txt
# CMD ["python", "application.py"]


FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install only required system packages and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements separately for better caching
COPY requirements.txt .

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy only your app code (not venv, cache, etc.)
COPY . .

# Run the app
CMD ["python", "application.py"]

