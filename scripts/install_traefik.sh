#!/bin/bash

TRAEFIK_VERSION=${1:-2.11.2}
TRAEFIK_ARCH=${2:-amd64}

# Download the tar.gz file
curl -L "https://github.com/traefik/traefik/releases/download/v${TRAEFIK_VERSION}/traefik_v${TRAEFIK_VERSION}_linux_${TRAEFIK_ARCH}.tar.gz" -o traefik.tar.gz

# Unzip the tar.gz file
tar xzf traefik.tar.gz

# Remove the tar.gz file
rm traefik.tar.gz
