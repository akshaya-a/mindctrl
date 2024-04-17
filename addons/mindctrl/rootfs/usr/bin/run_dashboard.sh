#!/usr/bin/env bashio
bashio::log.info "Starting dashboard script in $PWD"

ingress_entry=$(bashio::addon.ingress_entry)
bashio::log.info "ingress_entry: ${ingress_entry}"
# Am I nervous about these dependencies on dapr with a bunch of undocumented stuff? Yes.
# Am I going to do it anyway? Yes.
# https://github.com/dapr/dashboard/blob/a92b8cd20d97080f07518ced9a5e8d0a58168ad9/cmd/webserver.go#L148C47-L148C63
if [[ -n "$ingress_entry" ]]; then
    export SERVER_BASE_HREF="${ingress_entry}/dapr-dashboard/dapr-dashboard/"
    bashio::log.info "running dashboard with prefix $SERVER_BASE_HREF"
fi


dapr dashboard -a 0.0.0.0 -p 9999
