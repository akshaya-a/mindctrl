#!/usr/bin/env bashio

echo "Starting MLflow Deployment Server in $PWD"

if bashio::supervisor.ping; then
  bashio::log.info "Supervisor is running, setting config from supervisor"
  export OPENAI_API_KEY="$(bashio::config 'OPENAI_API_KEY')"
fi

bashio::log.info "Starting MLflow Deployment Server with Dapr..."
# https://github.com/dapr/dashboard/issues/195
s6-notifyoncheck dapr run --app-id deployments --app-port 5001 --app-protocol http \
  --enable-api-logging --enable-app-health-check --app-health-check-path /health --dapr-http-port 5501 -- \
  mlflow deployments start-server --config-path /services/deployments/route-config.yaml --port 5001 --host 0.0.0.0
