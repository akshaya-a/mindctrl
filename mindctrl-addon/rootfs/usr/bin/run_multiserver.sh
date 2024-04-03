#!/usr/bin/env bashio
bashio::log.level "all"

bashio::log.info "Starting multiserver script in $PWD"



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
export MLFLOW_DEPLOYMENTS_TARGET="http://0.0.0.0:5001"
# export PYTHONPATH="/usr/bin/multiserver"

bashio::log.info "Starting MLflow Tracking Server with Dapr..."
dapr run --app-id multiserver --app-port 5002 -- python3 -m uvicorn mindctrl.main:app --host 0.0.0.0 --port 5002
