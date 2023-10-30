#!/bin/bash

echo "Starting MLflow gateway"
key=$(python /config.py)
echo $key
export OPENAI_API_KEY=$key
mlflow gateway start --config-path /route-config.yaml --port 8099 --host 0.0.0.0
