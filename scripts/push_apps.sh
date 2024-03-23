#!/bin/bash
set -eu

cd ./mindctrl-addon/rootfs/usr/bin

echo "using target registry $K3D_REGISTRY_URL"

echo "Pushing the deployment-server image"
docker push $K3D_REGISTRY_URL/deployments:latest

echo "Pushing the multiserver image"
docker push $K3D_REGISTRY_URL/multiserver:latest

echo "Pushing the tracking image"
docker push $K3D_REGISTRY_URL/tracking:latest
