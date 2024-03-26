#!/usr/bin/env bashio

echo "RUNNING SERVER"
mlflow deployments start-server --config-path /route-config.yaml --port 5001 --host 0.0.0.0
