#!/usr/bin/env bash

VLFM_DIR=/coc/testnvme/nyokoyama3/fall_2024/implicit_memory_navigation/vlfm
VLFM_PYTHON=/coc/testnvme/nyokoyama3/miniconda3/envs/cobra_vlfm/bin/python

# Save current environment variables to disk
env_vars="/tmp/script_$(date +%Y%m%d_%H%M%S_%N).sh"
env | sed 's/^/export /' > $env_vars

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}

export tm=${tm:-"tmux"}
export session_name=${BLIP2_SESSION_NAME:-blip2_server_${RANDOM}}

# Create a new detached session with explicit socket path
echo "Creating new tmux session..."
$tm new-session -d -s "${session_name}" 2>/dev/null || true
$tm send-keys -t ${session_name}:0.0 "source ${env_vars} && cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m

echo "List of tmux windows:"
$tm ls

echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "${tm} attach-session -t ${session_name}"
