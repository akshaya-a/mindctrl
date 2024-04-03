#!/usr/bin/env bashio
bashio::log.level "all"

bashio::log.info "Starting MLflow tracking server script in $PWD"

# Do NOT cd - it breaks s6
# cd /usr/bin/tracking

dbpath="sqlite:////data/mydb.sqlite"

if [ ! -d "/data" ]; then
  bashio::log.info "/data does not exist, so assuming this is a test and placing locally"
  dbpath="sqlite:///mydb.sqlite"
fi

bashio::log.info "dbpath: ${dbpath}"


bashio::log.info "Starting MLflow Tracking Server with Dapr..."
export MLFLOW_DEPLOYMENTS_TARGET="http://0.0.0.0:5001"

bashio::log.info "Querying MLflow gateway health: ${MLFLOW_DEPLOYMENTS_TARGET}/health"
curl -i "${MLFLOW_DEPLOYMENTS_TARGET}/health"

s6-notifyoncheck dapr run --app-id tracking --app-port 5000 --app-protocol http --dapr-http-port 5500 -- \
  mlflow server \
  --backend-store-uri ${dbpath} \
  --artifacts-destination /share/tracking/mlflow-hass \
  --host 0.0.0.0 \
  --port 5000
