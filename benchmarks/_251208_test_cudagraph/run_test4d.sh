



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOBID=1113095
export HEAD_NODE_IP=fs-mbz-gpu-136
bash "$SCRIPT_DIR/test4d.sh" --config "$SCRIPT_DIR/pp_config.sh"