# Use official Python image
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY telemetry_feed.py ./telemetry_feed.py
COPY speed_profile_calculator.py ./speed_profile_calculator.py

ENV BACKEND_URL=ws://rust-backend:8765
CMD ["python", "telemetry_feed.py"]
