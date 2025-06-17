#!/bin/bash

# Set environment variables
export GROQ_API_KEY="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"
export MONGO_URL="mongodb://localhost:27017"
export DB_NAME="test_database"

# Change to backend directory
cd /app/backend

# Start uvicorn server
/root/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1 --reload