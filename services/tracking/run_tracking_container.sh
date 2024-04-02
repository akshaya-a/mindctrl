#!/usr/bin/env bashio

bashio::log.info "Starting MLflow tracking server script in $PWD"

# Do NOT cd - it breaks s6
# cd /usr/bin/tracking

dbpath="sqlite:////data/mydb.sqlite"
: ${ARTIFACTS_DESTINATION:="/artifacts"}

if [ ! -d "/data" ]; then
  bashio::log.info "/data does not exist, so assuming this is a test and placing locally"
  dbpath="sqlite:///mydb.sqlite"
fi

bashio::log.info "dbpath: ${dbpath}"
bashio::log.info "artifactspath: ${ARTIFACTS_DESTINATION}"

mlflow server \
  --backend-store-uri ${dbpath} \
  --artifacts-destination ${ARTIFACTS_DESTINATION} \
  --host 0.0.0.0 \
  --port 5000
