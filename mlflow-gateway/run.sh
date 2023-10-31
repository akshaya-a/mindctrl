#!/bin/bash

echo "Starting MLflow gateway"

key=$(python /config.py)
export OPENAI_API_KEY=$key

mlflow gateway start --config-path /route-config.yaml --port 5001 --host 0.0.0.0
