#!/usr/bin/env bashio

bashio::log.info "Starting traefik script in $PWD"


if bashio::supervisor.ping; then
    bashio::log.info "Supervisor is running, setting config from supervisor"
else
    bashio::log.info "Supervisor is not running, setting config from environment"
fi

export MLFLOW_TRACKING_URI="http://localhost:5000"
export MINDCTRL_SERVER_URI="http://localhost:5002"
export DAPR_DASHBOARD_URI="http://localhost:9999"

bashio::log.info "Using ${TRAEFIK_ALLOW_IP} as allowed source ips"

ingress_entry=$(bashio::addon.ingress_entry)
bashio::log.info "ingress_entry: ${ingress_entry}"
export HASS_INGRESS_ENTRY="${ingress_entry}"

bashio::log.info "Starting traefik..."
/traefik version
# TODO: until this is unified, keep in sync with testcontainer
/traefik --accesslog=true --accesslog.format=json --log.level=DEBUG --api=true --api.dashboard=true --api.insecure=true \
    --entrypoints.http.address=':80' \
    --ping=true \
    --providers.file.filename /.context/services/ingress/traefik-config.yaml
