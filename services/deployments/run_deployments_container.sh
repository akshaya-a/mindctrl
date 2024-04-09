#!/usr/bin/env bashio

echo "Running Replay Server..."

# Default values for MINDCTRL_REPLAY_DIR and MINDCTRL_RECORDING_DIR
export MINDCTRL_REPLAY_DIR=${MINDCTRL_REPLAY_DIR:="/replays"}
export MINDCTRL_RECORDING_DIR=${MINDCTRL_RECORDING_DIR:="/recordings"}
export MLFLOW_DEPLOYMENTS_CONFIG=${MLFLOW_DEPLOYMENTS_CONFIG:="/config/route-config.yaml"}

# If the environment variable MINDCTRL_CONFIG_REPLAY is set, set the --replay-path arg
if [ -z "$MINDCTRL_CONFIG_REPLAY" ]; then
    echo "MINDCTRL_CONFIG_REPLAY is not set. Running replay server in live mode"
    mindctrl serve
else
    mindctrl serve --replay
fi
