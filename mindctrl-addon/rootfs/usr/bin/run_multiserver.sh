#!/usr/bin/env bashio

bashio::log.info "Starting multiserver script in $PWD"



pyloc=$(which python3)
bashio::log.debug "pyloc: ${pyloc}"

export GIT_PYTHON_REFRESH=quiet


if bashio::supervisor.ping; then
    bashio::log.info "Supervisor is running, setting config from supervisor"
    response=$(bashio::api.supervisor GET "/addons/self/options/config" false)
    bashio::log.info "USING CONFIG: ${response}"

    export STORE__STORE_TYPE="$(bashio::config 'STORE__STORE_TYPE')"
    export STORE__USER="$(bashio::config 'STORE__USER')"
    export STORE__PASSWORD="$(bashio::config 'STORE__PASSWORD')"
    export STORE__ADDRESS="$(bashio::config 'STORE__ADDRESS')"
    export STORE__PORT="$(bashio::config 'STORE__PORT')"
    export STORE__DATABASE="$(bashio::config 'STORE__DATABASE')"

    export EVENTS__EVENTS_TYPE="$(bashio::config 'EVENTS__EVENTS_TYPE')"
    export EVENTS__BROKER="$(bashio::config 'EVENTS__BROKER')"
    export EVENTS__PORT="$(bashio::config 'EVENTS__PORT')"
    export EVENTS__USERNAME="$(bashio::config 'EVENTS__USERNAME')"
    export EVENTS__PASSWORD="$(bashio::config 'EVENTS__PASSWORD')"

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

bashio::log.info "Starting mindctrl server with Dapr..."
dapr run --log-level warn --app-id multiserver --app-port 5002 -- python3 -m uvicorn mindctrl.main:app --host 0.0.0.0 --port 5002
