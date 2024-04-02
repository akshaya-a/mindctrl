#!/usr/bin/env bashio

echo "Starting multiserver script in $PWD"

pyloc=$(which python3)
echo "pyloc: ${pyloc}"

export GIT_PYTHON_REFRESH=quiet

#export PYTHONPATH="/usr/bin/multiserver"

echo "Starting mindctrl Server..."
python3 -m uvicorn mindctrl.main:app --host 0.0.0.0 --port 5002
# python3 -m uvicorn main:app --host 0.0.0.0 --port 5002
