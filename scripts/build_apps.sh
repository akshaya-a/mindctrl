#!/bin/bash
set -eu

cd ./mindctrl-addon/rootfs/usr/bin

echo "using target registry $K3D_REGISTRY_URL"

echo "Building the deployments image"
docker build --pull --progress=plain --rm \
-f "./deployments/Dockerfile" \
-t $K3D_REGISTRY_URL/deployments:latest \
"./deployments" # context


echo "Building the multiserver image"
docker build --pull --progress=plain --rm \
-f "./multiserver/Dockerfile" \
-t $K3D_REGISTRY_URL/multiserver:latest \
"./multiserver" # context

echo "Building the tracking image"
docker build --pull --progress=plain --rm \
-f "./tracking/Dockerfile" \
-t $K3D_REGISTRY_URL/tracking:latest \
"./tracking" # context
