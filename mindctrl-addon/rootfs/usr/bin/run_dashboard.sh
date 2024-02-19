#!/usr/bin/env bashio

bashio::log.info "Starting dashboard script in $PWD"

dapr dashboard -a 0.0.0.0 -p 9999
