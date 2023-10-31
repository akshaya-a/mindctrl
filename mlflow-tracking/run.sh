#!/usr/bin/env bashio

bashio::log.info "Starting MLflow tracking server script"

# repo_slug -> repo-slug!
export MLFLOW_GATEWAY_URI="http://05657f25-mlflowgateway:5001"

bashio::log.info "Querying MLflow gateway health: ${MLFLOW_GATEWAY_URI}/health"
curl -i "${MLFLOW_GATEWAY_URI}/health"

mlflow server \
  --backend-store-uri sqlite:///data/mydb.sqlite \
  --artifacts-destination /share/tracking/mlflow-hass \
  --host 0.0.0.0 \
  --port 5000
