#!/usr/bin/env bashio

bashio::log.info "Starting multiserver script in $PWD"

# cd /usr/bin/multiserver


pyloc=$(which python3)
bashio::log.debug "pyloc: ${pyloc}"

export GIT_PYTHON_REFRESH=quiet


if bashio::supervisor.ping; then
    bashio::log.info "Supervisor is running, setting config from supervisor"
    export MQTT_BROKER="$(bashio::config 'MQTT_BROKER')"
    export MQTT_PORT="$(bashio::config 'MQTT_PORT')"
    export MQTT_USERNAME="$(bashio::config 'MQTT_USERNAME')"
    export MQTT_PASSWORD="$(bashio::config 'MQTT_PASSWORD')"

    export POSTGRES_USER="$(bashio::config 'POSTGRES_USER')"
    export POSTGRES_PASSWORD="$(bashio::config 'POSTGRES_PASSWORD')"
    export POSTGRES_ADDRESS="$(bashio::config 'POSTGRES_ADDRESS')"
    export POSTGRES_PORT="$(bashio::config 'POSTGRES_PORT')"

    export OPENAI_API_KEY="$(bashio::config 'OPENAI_API_KEY')"
else
    bashio::log.info "Supervisor is not running, setting config from environment"
    printenv
fi



ingress_entry=$(bashio::addon.ingress_entry)
bashio::log.info "ingress_entry: ${ingress_entry}"

notifyfd=$(</etc/s6-overlay/s6-rc.d/multiserver/notification-fd)
bashio::log.info "setting notification fd to ${notifyfd}"
export NOTIFY_FD="${notifyfd}"

export MLFLOW_TRACKING_URI="http://0.0.0.0:5000"
export PYTHONPATH="/usr/bin/multiserver"

if [[ -z "$DAPR_MODE" ]]; then
  export DAPR_MODE="$(bashio::config 'DAPR_MODE' || echo 'false')"
fi
if [ "$DAPR_MODE" = "true" ]; then
    bashio::log.info "Starting MLflow Tracking Server with Dapr so exiting..."
    dapr run --app-id multiserver --app-port 5002 -- \
        python3 -m uvicorn main:app --host 0.0.0.0 --port 5002

else
    bashio::log.info "Starting MLflow Tracking Server without Dapr..."

    python3 -m uvicorn main:app --host 0.0.0.0 --port 5002
fi
