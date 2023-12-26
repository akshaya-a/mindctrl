#!/usr/bin/env bashio

bashio::log.info "Starting multiserver script"

# repo_slug -> repo-slug!
export MLFLOW_TRACKING_URI="http://0.0.0.0:5000"

notifyfd=$(</etc/s6-overlay/s6-rc.d/multiserver/notification-fd)
bashio::log.info "setting notification fd to ${notifyfd}"
export NOTIFY_FD="${notifyfd}"

# bashio::log.info "Activating server env"
# source /usr/local/serverenv/bin/activate

pyloc=$(which python3)
bashio::log.info "pyloc: ${pyloc}"

export MQTT_BROKER="$(bashio::config 'MQTT_BROKER')"
export MQTT_PORT="$(bashio::config 'MQTT_PORT')"
export MQTT_USERNAME="$(bashio::config 'MQTT_USERNAME')"
export MQTT_PASSWORD="$(bashio::config 'MQTT_PASSWORD')"

export GIT_PYTHON_REFRESH=quiet

# change directory, otherwise uvicorn will not find multiserver.py
cd /usr/bin
python3 -m uvicorn multiserver.main:app --host 0.0.0.0 --port 5002
