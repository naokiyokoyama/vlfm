#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

VLFM_DIR=~/implicit_memory_navigation/vlfm
VLFM_PYTHON=/coc/testnvme/nyokoyama3/miniconda3/envs/cobra_vlfm/bin/python
COBRA_DIR=~/implicit_memory_navigation/cobra
COBRA_PYTHON=${COBRA_PYTHON:-/coc/testnvme/nyokoyama3/miniconda3/envs/cobra/bin/python}

# Save current environment variables to disk
env_vars="/tmp/script_$(date +%Y%m%d_%H%M%S_%N).sh"
env | sed 's/^/export /' > $env_vars

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export COBRA_PORT=${COBRA_PORT:-12185}

export tm=${tm:-"tmux"}
export session_name=${COBRA_SESSION_NAME:-cobra_server_${RANDOM}}

# Create a new detached session with explicit socket path
echo "Creating new tmux session..."
$tm new-session -d -s "${session_name}" 2>/dev/null || true
cmd="source ${env_vars} && cd ${COBRA_DIR} && ${COBRA_PYTHON} cobra/models/cobra_server.py --checkpoint ${COBRA_CKPT}"
$tm send-keys -t ${session_name} "$cmd --port ${COBRA_PORT}" C-m

echo "List of tmux windows:"
$tm ls

echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "${tm} attach-session -t ${session_name}"
