#!/bin/bash
# azure_deploy/startup.sh

# Fail fast
set -e

# Default PORT if not provided (Azure will inject PORT)
PORT="${PORT:-8501}"

echo "Starting Sentinel Streamlit app on port ${PORT}..."

cd /app

# You can log the mode if you ever need it
export SENTINEL_ENV="prod"

streamlit run sentinel_app/app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0