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
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export SAM_PORT=${SAM_PORT:-12183}
export YOLOV7_PORT=${YOLOV7_PORT:-12184}
export COBRA_PORT=${COBRA_PORT:-12185}

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

# Split the window into 2 or 3 panes
$tm split-window -v -t ${session_name}:0
if $has_o; then
    $tm split-window -v -t ${session_name}:0.0
fi

# Resize the panes so they are all equal size
$tm select-layout -t ${session_name}:0 even-vertical

# Run commands in each pane
$tm send-keys -t ${session_name}:0.0 "source ${env_vars} && cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m
$tm send-keys -t ${session_name}:0.1 "source ${env_vars} && cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m
if $has_o; then
    $tm send-keys -t ${session_name}:0.2 "cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
fi

echo "List of tmux windows:"
$tm ls

echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "${tm} attach-session -t ${session_name}"
