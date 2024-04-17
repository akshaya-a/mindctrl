#!/usr/bin/env bash

echo "Running mindctrl devcontainer entrypoint..."

echo "running original bootstrap"
bash devcontainer_bootstrap

echo "installing uv"
python3 -m pip install --upgrade uv

if [ -d "./.venv" ]; then
    echo ".venv does exist."
else
    echo "Create venv"
    uv venv
fi

echo "Activating venv"
source ./.venv/bin/activate

echo "install source"
uv pip install -e ./python

echo "install requirements"
uv pip install -r ./tests/test-requirements.txt


exec "$@"
