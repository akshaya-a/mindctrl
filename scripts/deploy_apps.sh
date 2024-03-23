#!/bin/bash
set -eu

cd ./mindctrl-addon/rootfs/usr/bin/deploy

echo "using target registry $K3D_REGISTRY_URL"

envsubst < deployments.yaml | kubectl apply -f -
envsubst < tracking.yaml | kubectl apply -f -
envsubst < multiserver.yaml | kubectl apply -f -
