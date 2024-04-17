#!/usr/bin/env bash

echo "Running mindctrl devcontainer entrypoint..."

echo "running original bootstrap"
bash devcontainer_bootstrap

if [ -d "./.venv" ]; then
    echo ".venv does exist."
else
    echo "Create venv"
    python3 -m venv ./.venv
fi

echo "Activating venv"
source ./.venv/bin/activate
python -m pip install --upgrade pip

echo "install source"
python -m pip install -e ./python

echo "install requirements"
python -m pip install -r ./tests/test-requirements.txt


exec "$@"
