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
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export SAM_PORT=${SAM_PORT:-12183}
export YOLOV7_PORT=${YOLOV7_PORT:-12184}
export COBRA_PORT=${COBRA_PORT:-12185}

session_name=vlm_servers_${RANDOM}
# Set the desired socket directory
echo "Directory path for tmux: $TMUX_TMPDIR"

# Ensure the directory exists
mkdir -p "$TMUX_TMPDIR"

# Construct the expected socket path
expected_socket_path="$TMUX_TMPDIR/tmux-$(id -u)/default"
mkdir -p "$(dirname "$expected_socket_path")"
chmod 700  "$(dirname "$expected_socket_path")"

# Function to check if a tmux server is actually running
is_tmux_running() {
    tmux list-sessions -F "#{socket_path}" >/dev/null 2>&1
    return $?
}

# Function to safely clean up stale socket
cleanup_stale_socket() {
    local socket_path="$1"
    if [ -S "$socket_path" ]; then
        echo "Found stale socket at $socket_path, removing it..."
        rm -f "$socket_path"
    fi
}

# Try to get current socket path
curr_socket_path=$(tmux display -pF '#{socket_path}' 2>/dev/null || echo "")
echo "Current tmux socket path: ${curr_socket_path:-"(no server running)"}"

# Check for and handle stale socket
if ! is_tmux_running; then
    cleanup_stale_socket "$expected_socket_path"
elif [ -n "$curr_socket_path" ] && [ "$curr_socket_path" != "$expected_socket_path" ]; then
    echo "Found tmux server using incorrect socket path: $curr_socket_path"
    echo "Expected path: $expected_socket_path"
    echo "Killing tmux server to allow use of correct path..."
    tmux kill-server
    cleanup_stale_socket "$expected_socket_path"
fi

# Create a new detached session with explicit socket path
echo "Creating new tmux session..."
tmux -S "$expected_socket_path" new-session -d -s "${session_name}" 2>/dev/null || true

# Verify the final socket path
final_socket_path=$(tmux -S "$expected_socket_path" display -pF '#{socket_path}' 2>/dev/null || echo "(failed to create session)")
echo "Deploying at tmux socket path: $final_socket_path"

# Split the window vertically
tmux -S "$expected_socket_path" split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux -S "$expected_socket_path" split-window -h -t ${session_name}:0.0
tmux -S "$expected_socket_path" split-window -h -t ${session_name}:0.2

# Check if the -c flag is set
has_c=false
while getopts "c" flag; do
    case "${flag}" in
        c) has_c=true;;
    esac
done

# Run commands in each pane
#tmux send-keys -t ${session_name}:0.0 "cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
if $has_c; then
    tmux -S "$expected_socket_path" send-keys -t ${session_name}:0.1 "source ${env_vars} && cd ${COBRA_DIR} && ${COBRA_PYTHON} cobra/models/cobra_server.py --checkpoint ${COBRA_CKPT} --port ${COBRA_PORT}" C-m
else
    tmux -S "$expected_socket_path" send-keys -t ${session_name}:0.1 "source ${env_vars} && cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
fi
tmux -S "$expected_socket_path" send-keys -t ${session_name}:0.2 "source ${env_vars} && cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m
tmux -S "$expected_socket_path" send-keys -t ${session_name}:0.3 "source ${env_vars} && cd ${VLFM_DIR} && ${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

echo "List of tmux windows:"
tmux -S "$expected_socket_path" ls

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux -S ${final_socket_path} attach-session -t ${session_name}"
