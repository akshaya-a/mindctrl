#!/usr/bin/env bashio

bashio::log.info "Starting MLflow tracking server script"

# repo_slug -> repo-slug!
export MLFLOW_GATEWAY_URI="http://0.0.0.0:5001"

bashio::log.info "Querying MLflow gateway health: ${MLFLOW_GATEWAY_URI}/health"
curl -i "${MLFLOW_GATEWAY_URI}/health"

dbpath="sqlite:////data/mydb.sqlite"

if [ ! -d "/data" ]; then
  bashio::log.info "/data does not exist, so assuming this is a test and placing locally"
  dbpath="sqlite:///mydb.sqlite"
fi

mlflow server \
  --backend-store-uri ${dbpath} \
  --artifacts-destination /share/tracking/mlflow-hass \
  --host 0.0.0.0 \
  --port 5000
