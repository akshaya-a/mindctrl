#!/usr/bin/env bashio

echo "Starting MLflow Deployment Server in $PWD"

if bashio::supervisor.ping; then
  bashio::log.info "Supervisor is running, setting config from supervisor"
  export OPENAI_API_KEY="$(bashio::config 'OPENAI_API_KEY')"
fi

# cd /usr/bin/deployment-server
# if the environment variable DAPR_MODE is set to true, invoke dapr cli instead
if [[ -z "$DAPR_MODE" ]]; then
  export DAPR_MODE="$(bashio::config 'DAPR_MODE' || echo 'false')"
fi
if [ "$DAPR_MODE" = "true" ]; then
  bashio::log.info "Starting MLflow Deployment Server with Dapr..."
  s6-notifyoncheck dapr run --app-id deployment-server --app-port 5001 --app-protocol http \
    --enable-app-health-check --app-health-check-path --dapr-http-port 5501 -- \
    mlflow deployments start-server --config-path /usr/bin/deployment-server/route-config.yaml --port 5001 --host 0.0.0.0
else
  bashio::log.info "Starting MLflow Deployment Server without Dapr"
  # rootfs/route-config.yaml
  s6-notifyoncheck mlflow deployments start-server --config-path ./route-config.yaml --port 5001 --host 0.0.0.0
fi
