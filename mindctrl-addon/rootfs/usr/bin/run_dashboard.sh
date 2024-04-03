#!/usr/bin/env bashio
bashio::log.level "all"

bashio::log.info "Starting dashboard script in $PWD"

dapr dashboard -a 0.0.0.0 -p 9999
