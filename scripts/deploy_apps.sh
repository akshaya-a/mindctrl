#!/bin/bash
set -eu

cd ./mindctrl-addon/rootfs/usr/bin

echo "using target registry $K3D_REGISTRY_URL"

envsubst < /deployments.yml | kubectl apply -f -
envsubst < /tracking.yml | kubectl apply -f -
envsubst < /multiserver.yml | kubectl apply -f -
