#!/usr/bin/env bash
set -e

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start FastAPI with uvicorn on Azure's assigned $PORT
uvicorn app:app --host 0.0.0.0 --port $PORT
