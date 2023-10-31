#!/usr/bin/env bashio

echo "Starting MLflow server"

echo "Calling supervisor"
addons = $(curl -X GET -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" -H "Content-Type: application/json" http://supervisor/addons)
echo $addons

bashio::supervisor.addons

bashio::api.supervisor GET /addons

# TODO: convert this to Supervisor REPO_SLUG format via addon discovery
export MLFLOW_GATEWAY_URI="http://localhost:5001"

mlflow server \
  --backend-store-uri sqlite:///data/mydb.sqlite \
  --artifacts-destination /share/tracking/mlflow-hass \
  --host 0.0.0.0 \
  --port 5000
