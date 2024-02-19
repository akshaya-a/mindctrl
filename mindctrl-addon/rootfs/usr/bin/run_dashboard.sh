#!/usr/bin/env bashio

bashio::log.info "Starting dashboard script in $PWD"

# cd /usr/bin/multiserver

dapr dashboard -a 0.0.0.0 -p 9999
