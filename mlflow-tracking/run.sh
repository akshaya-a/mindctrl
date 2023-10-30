#!/bin/bash

echo "Starting MLflow server"
mlflow server \
  --backend-store-uri sqlite:///data/mydb.sqlite \
  --artifacts-destination /share/tracking/mlflow-hass \
  --host 0.0.0.0 \
  --port 8099
