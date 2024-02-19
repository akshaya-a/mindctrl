#!/usr/bin/env bashio

echo "Starting MLflow Deployment Server in $PWD"

if bashio::supervisor.ping; then
  bashio::log.info "Supervisor is running, setting config from supervisor"
  export OPENAI_API_KEY="$(bashio::config 'OPENAI_API_KEY')"
fi

bashio::log.info "Starting MLflow Deployment Server with Dapr..."
s6-notifyoncheck dapr run --app-id deployment-server --app-port 5001 --app-protocol http \
  --enable-app-health-check --app-health-check-path /health --dapr-http-port 5501 -- \
  mlflow deployments start-server --config-path /usr/bin/deployment-server/route-config.yaml --port 5001 --host 0.0.0.0
