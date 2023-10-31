#!/bin/bash

echo "Starting MLflow gateway"
key=$(python /config.py)
echo $key
export OPENAI_API_KEY=$key
export PYTHONPATH="/"
mlflow gateway start --config-path /route-config.yaml --port 5001 --host 0.0.0.0
