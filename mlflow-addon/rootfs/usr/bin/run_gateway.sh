#!/usr/bin/env bashio

echo "Starting MLflow gateway"

export OPENAI_API_KEY="$(bashio::config 'OPENAI_API_KEY')"

# rootfs/route-config.yaml
s6-notifyoncheck mlflow gateway start --config-path /route-config.yaml --port 5001 --host 0.0.0.0
