#!/bin/bash
set -eu

cd ./services

echo "using target registry $K3D_REGISTRY_URL"

echo "Pushing the deployments image"
docker push $K3D_REGISTRY_URL/deployments:latest

echo "Pushing the multiserver image"
docker push $K3D_REGISTRY_URL/multiserver:latest

echo "Pushing the tracking image"
docker push $K3D_REGISTRY_URL/tracking:latest
