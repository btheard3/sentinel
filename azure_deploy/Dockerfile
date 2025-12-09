# azure_deploy/Dockerfile

FROM python:3.12-slim

# --- Basics & env hygiene ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Azure-style PORT env (used by startup.sh)
ENV PORT=8501

# --- Workdir ---
WORKDIR /app

# --- Install system deps (if you ever add numpy/scipy, etc.) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
# If your requirements file has a different name/path, adjust this line.
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project code, models, data, etc. ---
COPY . .

# Make sure startup script is executable
RUN chmod +x azure_deploy/startup.sh

# --- Streamlit / networking config ---
EXPOSE 8501

# Azure will set $PORT; locally we’ll also use it.
CMD ["bash", "azure_deploy/startup.sh"]