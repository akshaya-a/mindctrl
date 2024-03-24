#!/bin/bash
set -eu

cd ./mindctrl-addon/rootfs/usr/bin/deploy

echo "using target registry $K3D_REGISTRY_URL"
echo "using registry for pull $REGISTRY_NAME:$REGISTRY_PORT"

envsubst < deployments.yaml > deployments-resolved.yaml
cat deployments-resolved.yaml
kubectl apply -f deployments-resolved.yaml

envsubst < tracking.yaml > tracking-resolved.yaml
cat tracking-resolved.yaml
kubectl apply -f tracking-resolved.yaml

envsubst < multiserver.yaml > multiserver-resolved.yaml
cat multiserver-resolved.yaml
kubectl apply -f multiserver-resolved.yaml
