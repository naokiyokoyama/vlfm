#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

VLFM_DIR=~/implicit_memory_navigation/vlfm
VLFM_PYTHON=/coc/testnvme/nyokoyama3/miniconda3/envs/cobra_vlfm/bin/python

# Save current environment variables to disk
env_vars="/tmp/script_$(date +%Y%m%d_%H%M%S_%N).sh"
env | sed 's/^/export /' > $env_vars

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export OWLV2_PORT=${OWLV2_PORT:-12181}
export SAM_PORT=${SAM_PORT:-12183}
export YOLO_PORT=${YOLO_PORT:-12184}

export tm=${tm:-"tmux"}
export session_name=${VLM_SESSION_NAME:-vlm_servers_${RANDOM}}

# Create a new detached session with explicit socket path
echo "Creating new tmux session..."
$tm new-session -d -s "${session_name}" 2>/dev/null || true

# Check if the -o flag is set
has_o=false
while getopts "o" flag; do
    case "${flag}" in
        o) has_o=true;;
    esac
done

# Split the window into 2 panes
$tm split-window -v -t ${session_name}:0

# Run commands in each pane
$tm send-keys -t ${session_name}:0.0 "source ${env_vars} && cd ${VLFM_DIR} && while true; do ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}; done" C-m
$tm send-keys -t ${session_name}:0.1 "source ${env_vars} && cd ${VLFM_DIR} && while true; do ${VLFM_PYTHON} -m vlfm.vlm.yolo --port ${YOLO_PORT}; done" C-m
if $has_o; then
    $tm split-window -v -t ${session_name}:0.1
    $tm send-keys -t ${session_name}:0.2 "source ${env_vars} && cd ${VLFM_DIR} && while true; do ${VLFM_PYTHON} -m vlfm.vlm.owlv2 --port ${OWLV2_PORT}; done" C-m
fi

# Resize the panes so they are all equal size
$tm select-layout -t ${session_name}:0 even-vertical

echo "List of tmux windows:"
$tm ls

echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "${tm} attach-session -t ${session_name}"
