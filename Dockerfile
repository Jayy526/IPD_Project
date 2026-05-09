FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000 8501

# Note: In a real production deployment, you'd likely split these into two containers (docker-compose).
# For simplicity here, we can start both using a bash script or just run the backend.
# To run backend: CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
# To run frontend: CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

CMD ["bash", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
