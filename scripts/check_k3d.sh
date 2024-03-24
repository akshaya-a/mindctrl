#!/bin/bash
set -eu

kubectl get nodes
docker ps -f name=$REGISTRY_NAME
k3d registry list
nslookup k3d-$REGISTRY_NAME

# https://k3d.io/v5.2.0/usage/registries/#testing-your-registry
echo "Testing registry at $K3D_REGISTRY_URL"
docker pull nginx:latest
docker tag nginx:latest $K3D_REGISTRY_URL/nginx:latest
docker push $K3D_REGISTRY_URL/nginx:latest

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-test-registry
  labels:
    app: nginx-test-registry
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx-test-registry
  template:
    metadata:
      labels:
        app: nginx-test-registry
    spec:
      containers:
      - name: nginx-test-registry
        image: k3d-registry.localhost:12345/nginx:latest
        ports:
        - containerPort: 80
EOF

echo "Waiting for deployment to be ready..."
# sleep for 10 seconds to give the deployment time to start
sleep 10
kubectl get pods -l "app=nginx-test-registry"

echo "Cluster is validated"
