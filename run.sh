#!/bin/bash
# Quick start for the Living Image Crossfade Test Harness

# Load .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "Starting Living Image Crossfade Test Harness..."
echo "Open http://localhost:8000 in your browser"
echo ""

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
