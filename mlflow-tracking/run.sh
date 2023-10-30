#!/bin/bash

echo "Starting MLflow server"
mlflow server \
  --backend-store-uri sqlite:///mydb.sqlite \
  --artifacts-destination ./mlruns \
  --host 0.0.0.0 \
  --port 8099
