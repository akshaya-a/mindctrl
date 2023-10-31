#!/usr/bin/env bashio

echo "Starting MLflow server script"

bashio::log.info "Querying addons for gateway"
bashio::addons.installed

bashio::log.info "Starting MLflow server"

# TODO: convert this to Supervisor REPO_SLUG format via addon discovery
export MLFLOW_GATEWAY_URI="http://localhost:5001"

mlflow server \
  --backend-store-uri sqlite:///data/mydb.sqlite \
  --artifacts-destination /share/tracking/mlflow-hass \
  --host 0.0.0.0 \
  --port 5000
