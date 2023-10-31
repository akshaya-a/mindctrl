#!/bin/bash

echo "Starting MLflow gateway"

CONFIG_PATH=/data/options.json
export OPENAI_API_KEY="$(bashio::config 'OPENAI_API_KEY')"

mlflow gateway start --config-path /route-config.yaml --port 5001 --host 0.0.0.0
