set -e

# -----------------------------
# Command Line Argument Parsing
# -----------------------------
LIST_ONLY=0
LIST_RUNS=0
LIST_RUNS_FULL=0
DRY_RUN=0
SKIP_PARTIAL=0
SKIP_STARTED=0
SKIP_COMPLETED=0
SKIP_OOM=1  # Skip OOM cases by default
MONITOR_STUCK=0
DEBUG_ONLY=0

# Configuration file path
CONFIG_FILE=""

# Filter variables
FILTER_MODEL_PATH=""
FILTER_ATB=""
FILTER_NLAYERS=""
FILTER_N=""
FILTER_BS=""
FILTER_MB=""
FILTER_TOKS=""
FILTER_MODE=""
FILTER_CP=""
FILTER_PP=""
FILTER_TP=""
FILTER_PRIORITY_GE=""
FILTER_PRIORITY_LE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ls)
            LIST_ONLY=1
            shift
            ;;
        --ls-runs)
            LIST_RUNS=1
            shift
            ;;
        --ls-runs-full)
            LIST_RUNS_FULL=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-partial)
            SKIP_PARTIAL=1
            shift
            ;;
        --skip-started)
            SKIP_STARTED=1
            shift
            ;;
        --skip-completed)
            SKIP_COMPLETED=1
            shift
            ;;
        --no-skip-oom)
            SKIP_OOM=0
            shift
            ;;
        --monitor-stuck)
            MONITOR_STUCK=1
            shift
            ;;
        --debug-only)
            DEBUG_ONLY=1
            shift
            ;;
        --filter-model_path)
            FILTER_MODEL_PATH="$2"
            shift 2
            ;;
        --filter-atb)
            FILTER_ATB="$2"
            shift 2
            ;;
        --filter-nlayers)
            FILTER_NLAYERS="$2"
            shift 2
            ;;
        --filter-N)
            FILTER_N="$2"
            shift 2
            ;;
        --filter-bs)
            FILTER_BS="$2"
            shift 2
            ;;
        --filter-mb)
            FILTER_MB="$2"
            shift 2
            ;;
        --filter-toks)
            FILTER_TOKS="$2"
            shift 2
            ;;
        --filter-mode)
            FILTER_MODE="$2"
            shift 2
            ;;
        --filter-cp)
            FILTER_CP="$2"
            shift 2
            ;;
        --filter-pp)
            FILTER_PP="$2"
            shift 2
            ;;
        --filter-tp)
            FILTER_TP="$2"
            shift 2
            ;;
        --filter-priority-ge)
            FILTER_PRIORITY_GE="$2"
            shift 2
            ;;
        --filter-priority-le)
            FILTER_PRIORITY_LE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --config CONFIG_FILE [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --config FILE   Specify config file to use"
            echo ""
            echo "Options:"
            echo "  --ls            List configs that would be run and exit"
            echo "  --ls-runs       List configs and show existing run folders, then exit"
            echo "  --ls-runs-full  List ALL configs (including completed) and show existing run folders, then exit"
            echo "  --dry-run       Show what would be run but don't execute"
            echo "  --skip-partial  Skip configs that already have partial results (benchmark.raw.jsonl)"
            echo "  --skip-started  Skip configs that have already been started (have README.md)"
            echo "  --skip-completed Skip configs that have been completed with sufficient samples (>= MAX_SAMPLE_ID)"
            echo "  --no-skip-oom   Don't skip configs that have OOM errors (by default OOM cases are skipped)"
            echo "  --monitor-stuck Enable job monitoring to detect stuck processes (disabled by default)"
            echo "  --debug-only    Only run configs with EXPERIMENT__DEBUG=1 (by default debug configs are filtered out)"
            echo ""
            echo "Filter Options (comma-separated values for OR condition):"
            echo "  --filter-model_path VALUE   Filter by model path"
            echo "  --filter-atb VALUE          Filter by attention breakpoint"
            echo "  --filter-nlayers VALUE      Filter by number of layers"
            echo "  --filter-N VALUE            Filter by number of nodes"
            echo "  --filter-bs VALUE           Filter by batch size"
            echo "  --filter-mb VALUE           Filter by microbatch size"
            echo "  --filter-toks VALUE         Filter by number of tokens"
            echo "  --filter-mode VALUE         Filter by mode"
            echo "  --filter-cp VALUE           Filter by CP size"
            echo "  --filter-pp VALUE           Filter by PP size"
            echo "  --filter-tp VALUE           Filter by TP size"
            echo "  --filter-priority-ge VALUE  Filter by priority >= VALUE (default priority: 10)"
            echo "  --filter-priority-le VALUE  Filter by priority <= VALUE (default priority: 10)"
            echo ""
            echo "Per-Config Environment Variables (set in env_var field of each config line):"
            echo "  CHECK_TOTAL_BS=N        Skip config if batch_size * microbatch_size != N"
            echo ""
            echo "Examples:"
            echo "  $0 --config config_sweep_memory.config.v3.sh --filter-N 16,32 --filter-mode wlbllm,d2"
            echo "  $0 --config config_sweep_memory.config.v3.sh --ls"
            echo "  $0 --config my_custom_config.sh --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --config CONFIG_FILE [OPTIONS]"
            echo "Use --help for more information"
            exit 1
            ;;
    esac
done

# Validate that config file is specified
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: --config argument is required"
    echo "Usage: $0 --config CONFIG_FILE [OPTIONS]"
    echo "Use --help for more information"
    exit 1
fi

# Validate that config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist"
    exit 1
fi

# export JOBID=${JOBID:-710588}
TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST

# Function to check if a value matches any of the comma-separated filter values
# Usage: matches_filter "actual_value" "filter_string"
# Returns 0 (true) if match found or filter is empty, 1 (false) otherwise
matches_filter() {
    local actual_value="$1"
    local filter_string="$2"
    
    # If filter is empty, accept all values
    if [ -z "$filter_string" ]; then
        return 0
    fi
    
    # Split filter string by comma and check each value
    IFS=',' read -ra filter_values <<< "$filter_string"
    for filter_value in "${filter_values[@]}"; do
        # Trim whitespace
        filter_value=$(echo "$filter_value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [ "$actual_value" = "$filter_value" ]; then
            return 0
        fi
    done
    
    return 1
}

# Function to check if a case matches all active filters
# Usage: case_matches_filters model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size env_var
case_matches_filters() {
    local model_path="$1"
    local attn_linear_breakpoint="$2"
    local num_layers="$3"
    local nnodes="$4"
    local batch_size="$5"
    local microbatch_size="$6"
    local num_tokens="$7"
    local mode="$8"
    local cp_size="$9"
    local pp_size="${10}"
    local tp_size="${11}"
    local env_var="${12}"
    
    # Check each filter - if any filter fails, return false
    if ! matches_filter "$model_path" "$FILTER_MODEL_PATH"; then
        return 1
    fi
    
    if ! matches_filter "$attn_linear_breakpoint" "$FILTER_ATB"; then
        return 1
    fi
    
    if ! matches_filter "$num_layers" "$FILTER_NLAYERS"; then
        return 1
    fi
    
    if ! matches_filter "$nnodes" "$FILTER_N"; then
        return 1
    fi
    
    if ! matches_filter "$batch_size" "$FILTER_BS"; then
        return 1
    fi
    
    if ! matches_filter "$microbatch_size" "$FILTER_MB"; then
        return 1
    fi
    
    if ! matches_filter "$num_tokens" "$FILTER_TOKS"; then
        return 1
    fi
    
    if ! matches_filter "$mode" "$FILTER_MODE"; then
        return 1
    fi
    
    if ! matches_filter "$cp_size" "$FILTER_CP"; then
        return 1
    fi
    
    if ! matches_filter "$pp_size" "$FILTER_PP"; then
        return 1
    fi
    
    if ! matches_filter "$tp_size" "$FILTER_TP"; then
        return 1
    fi
    
    # Check priority filters if specified
    if [ ! -z "$FILTER_PRIORITY_GE" ] || [ ! -z "$FILTER_PRIORITY_LE" ]; then
        local priority=$(extract_schedule_priority "$env_var")
        
        if [ ! -z "$FILTER_PRIORITY_GE" ]; then
            if [ "$priority" -lt "$FILTER_PRIORITY_GE" ]; then
                return 1
            fi
        fi
        
        if [ ! -z "$FILTER_PRIORITY_LE" ]; then
            if [ "$priority" -gt "$FILTER_PRIORITY_LE" ]; then
                return 1
            fi
        fi
    fi
    
    # All filters passed
    return 0
}

# Function to parse global variable overrides from config file
# Usage: parse_global_variable_overrides CONFIG_FILE
parse_global_variable_overrides() {
    local config_file="$1"
    
    if [ ! -f "$config_file" ]; then
        return 0  # No config file, nothing to parse
    fi
    
    echo "🔧 Parsing global variable overrides from config file..."
    
    # Read config file line by line looking for # $$VARIABLE=value patterns
    while IFS= read -r line; do
        # Skip empty lines and regular comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*$ ]] && continue
        
        # Check for global variable override pattern: # $$VARIABLE=value
        if [[ "$line" =~ ^[[:space:]]*#[[:space:]]*\$\$([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
            var_name="${BASH_REMATCH[1]}"
            var_value="${BASH_REMATCH[2]}"
            
            # Remove any trailing whitespace from value
            var_value=$(echo "$var_value" | sed 's/[[:space:]]*$//')
            
            # Export the variable
            export "$var_name"="$var_value"
            echo "🔧 Global override: $var_name=$var_value"
        fi
    done < "$config_file"
}

# Parse global variable overrides from config file before setting defaults
parse_global_variable_overrides "$CONFIG_FILE"


# Use EXPERIMENT_DISTS if set, otherwise fallback to default
if [ -z "${EXPERIMENT_DISTS}" ]; then
    dists=(
        "wlbllm 0.0"
    )
else
    # Check if EXPERIMENT_DISTS looks like bash array syntax
    if [[ "${EXPERIMENT_DISTS}" =~ ^[[:space:]]*\(.*\)[[:space:]]*$ ]]; then
        # Parse bash array syntax: ("item1" "item2" "item3")
        # Remove outer parentheses and leading/trailing whitespace
        array_content="${EXPERIMENT_DISTS#*\(}"
        array_content="${array_content%\)*}"
        array_content="${array_content# }"
        array_content="${array_content% }"
        
        # Use eval to safely parse the quoted strings into an array
        eval "dists=($array_content)"
    else
        # Fallback: Split EXPERIMENT_DISTS string into array (comma-separated)
        IFS=',' read -ra dists <<< "${EXPERIMENT_DISTS}"
    fi
fi

for sample_config in "${dists[@]}"; do
    echo sample_config="$sample_config" ";"
done




# Set OUTPUT_DIR_PREFIX with default value if not already set by config file
export OUTPUT_DIR_PREFIX=${OUTPUT_DIR_PREFIX:-/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v4-sweep-pp-34b}

export STATUS_FILE_PATH=${OUTPUT_DIR_PREFIX}/job-status.txt
mkdir -p ${OUTPUT_DIR_PREFIX}
touch ${STATUS_FILE_PATH}

# Function to analyze the status of a run directory
# Returns: STATUS:LINE_COUNT for SUCCESS/PARTIAL (e.g., SUCCESS:42, PARTIAL:15), or just STATUS for others
analyze_run_status() {
    local run_dir=$1
    
    # Check for success indicator
    if [ -f "$run_dir/benchmark.json" ]; then
        # Count lines in benchmark.raw.jsonl if it exists
        if [ -f "$run_dir/benchmark.raw.jsonl" ]; then
            line_count=$(wc -l < "$run_dir/benchmark.raw.jsonl" 2>/dev/null || echo "0")
            echo "SUCCESS:$line_count"
        else
            echo "SUCCESS:0"
        fi
        return
    fi
    
    # Check for partial success indicator
    if [ -f "$run_dir/benchmark.raw.jsonl" ]; then
        line_count=$(wc -l < "$run_dir/benchmark.raw.jsonl" 2>/dev/null || echo "0")
        echo "PARTIAL:$line_count"
        return
    fi
    
    # Check for OOM error in logs
    if [ -d "$run_dir/logs" ]; then
        # Use find to avoid shell globbing issues and limit scope
        if find "$run_dir/logs" -name "*.log" -type f -exec grep -l "OutOfMemoryError" {} + 2>/dev/null | head -1 | grep -q .; then
            echo "OOM"
            return
        fi
    fi
    
    # Check if run started (has README.md)
    if [ -f "$run_dir/README.md" ]; then
        # Check for MAYBE_STUCK condition using job liveness and log update times
        liveness_file="$run_dir/job_liveness.txt"
        
        if [ -f "$liveness_file" ]; then
            # Get the last liveness timestamp
            last_liveness_time=$(cat "$liveness_file" 2>/dev/null || echo "0")
            current_time=$(date +%s)
            liveness_age_seconds=$((current_time - last_liveness_time))
            
            # If liveness timestamp is recent (within 30 seconds), job is likely still alive
            if [ $liveness_age_seconds -lt 30 ]; then
                # Job is alive, check log update times
                if [ -d "$run_dir/logs" ]; then
                    # Find the most recent log file update
                    latest_log_time=$(find "$run_dir/logs" -name "*.log" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
                    
                    if [ ! -z "$latest_log_time" ]; then
                        log_age_seconds=$(echo "$current_time - $latest_log_time" | bc -l)
                        log_age_minutes=$(echo "$log_age_seconds / 60" | bc -l)
                        
                        # If logs haven't been updated for >5 minutes while job is alive, it's stuck
                        if (( $(echo "$log_age_minutes > 5" | bc -l) )); then
                            echo "MAYBE_STUCK:${log_age_minutes%.*}min"
                            return
                        fi
                    fi
                fi
                
                # Job is alive and logs are recent, or no logs yet
                echo "STARTED"
                return
            else
                # Job liveness is old, so job is probably dead - just started then died
                echo "STARTED"
                return
            fi
        else
            # No liveness file, fall back to basic STARTED status
            echo "STARTED"
            return
        fi
    fi
    
    echo "UNKNOWN"
}

# Function to get status emoji and description
get_status_display() {
    local status_info=$1
    
    # Parse status and line count if present
    if [[ "$status_info" == *":"* ]]; then
        local status="${status_info%:*}"
        local line_count="${status_info#*:}"
    else
        local status="$status_info"
        local line_count=""
    fi
    
    case $status in
        "SUCCESS")
            if [ ! -z "$line_count" ]; then
                echo "✅ SUCCESS (benchmark.json, $line_count lines in raw)"
            else
                echo "✅ SUCCESS (benchmark.json)"
            fi
            ;;
        "PARTIAL")
            if [ ! -z "$line_count" ]; then
                echo "🟡 PARTIAL (benchmark.raw.jsonl, $line_count lines)"
            else
                echo "🟡 PARTIAL (benchmark.raw.jsonl)"
            fi
            ;;
        "STARTED")
            echo "🟠 STARTED (README.md)"
            ;;
        "MAYBE_STUCK")
            if [ ! -z "$line_count" ]; then
                echo "🟡 MAYBE_STUCK (${line_count} since last log update, job alive)"
            else
                echo "🟡 MAYBE_STUCK (>5min since last log update, job alive)"
            fi
            ;;
        "OOM")
            echo "🔴 OOM (OutOfMemoryError in logs)"
            ;;
        "UNKNOWN")
            echo "❓ UNKNOWN"
            ;;
    esac
}

# Function to find existing run folders for a given config
# Usage: find_existing_runs mode nnodes num_tokens batch_size microbatch_size cp_size tp_size pp_size model_path_normalized
find_existing_runs() {
    local mode=$1
    local nnodes=$2
    local num_tokens=$3
    local batch_size=$4
    local microbatch_size=$5
    local cp_size=$6
    local tp_size=$7
    local pp_size=$8
    local model_path_normalized=$9
    
    # Construct the search pattern (ignoring datetime part)
    # Pattern: <datetime>.<mode>-n<node>-t<tokens>-b<batchsize>-mb<microbatchsize>-cp<cp_size>tp<tpsize>pp<pp_size>-<modelname>
    local search_pattern="*.${mode}-n${nnodes}-t${num_tokens}-b${batch_size}-mb${microbatch_size}-cp${cp_size}tp${tp_size}pp${pp_size}-${model_path_normalized}*"
    # echo "search_pattern: $search_pattern"
    
    # Search for matching directories
    if [ -d "$OUTPUT_DIR_PREFIX" ]; then
        find "$OUTPUT_DIR_PREFIX" -maxdepth 1 -type d -name "$search_pattern" 2>/dev/null | sort
    fi
}

# Function to check if a config has partial results (benchmark.raw.jsonl files)
config_has_partial_results() {
    local mode=$1
    local nnodes=$2
    local num_tokens=$3
    local batch_size=$4
    local microbatch_size=$5
    local cp_size=$6
    local tp_size=$7
    local pp_size=$8
    local model_path_normalized=$9
    
    local existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
    
    if [ -z "$existing_runs" ]; then
        return 1  # No existing runs, so no partial results
    fi
    
    # Check if any run has benchmark.raw.jsonl (partial results)
    while IFS= read -r run_dir; do
        if [ -z "$run_dir" ]; then continue; fi
        if [ -f "$run_dir/benchmark.raw.jsonl" ]; then
            return 0  # Found partial results
        fi
    done <<< "$existing_runs"
    
    return 1  # No partial results found
}

# Function to check if a config has already been started (has README.md but no results)
config_has_started() {
    local mode=$1
    local nnodes=$2
    local num_tokens=$3
    local batch_size=$4
    local microbatch_size=$5
    local cp_size=$6
    local tp_size=$7
    local pp_size=$8
    local model_path_normalized=$9
    
    local existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
    
    if [ -z "$existing_runs" ]; then
        return 1  # No existing runs, so not started
    fi
    
    # Check if any run has "STARTED" status (has README.md but no benchmark.json or benchmark.raw.jsonl)
    while IFS= read -r run_dir; do
        if [ -z "$run_dir" ]; then continue; fi
        local status=$(analyze_run_status "$run_dir")
        # Check if status is STARTED or MAYBE_STUCK (both indicate job was started)
        if [[ "$status" == "STARTED" ]] || [[ "$status" == "MAYBE_STUCK"* ]]; then
            return 0  # Found started run
        fi
    done <<< "$existing_runs"
    
    return 1  # No started runs found
}

# Function to check if a config has been completed with sufficient samples
# Returns 0 (true) if there's a successful run with >= MAX_SAMPLE_ID samples
config_is_completed_with_sufficient_samples() {
    local mode=$1
    local nnodes=$2
    local num_tokens=$3
    local batch_size=$4
    local microbatch_size=$5
    local cp_size=$6
    local tp_size=$7
    local pp_size=$8
    local model_path_normalized=$9
    
    local existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
    
    if [ -z "$existing_runs" ]; then
        return 1  # No existing runs, so not completed
    fi
    
    # Check if any run has benchmark.json (success) AND sufficient samples in benchmark.raw.jsonl
    while IFS= read -r run_dir; do
        if [ -z "$run_dir" ]; then continue; fi
        
        # Check if this run was successful (has benchmark.json)
        if [ -f "$run_dir/benchmark.json" ]; then
            # Check if benchmark.raw.jsonl exists and has sufficient samples
            if [ -f "$run_dir/benchmark.raw.jsonl" ]; then
                local line_count=$(wc -l < "$run_dir/benchmark.raw.jsonl" 2>/dev/null || echo "0")
                if [ "$line_count" -ge "$MAX_SAMPLE_ID" ]; then
                    return 0  # Found completed run with sufficient samples
                fi
            fi
        fi
    done <<< "$existing_runs"
    
    return 1  # No completed runs with sufficient samples found
}

# Function to check if a config has OOM errors
# Returns 0 (true) if there's any run with OOM status
config_has_oom() {
    local mode=$1
    local nnodes=$2
    local num_tokens=$3
    local batch_size=$4
    local microbatch_size=$5
    local cp_size=$6
    local tp_size=$7
    local pp_size=$8
    local model_path_normalized=$9
    
    local existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
    
    if [ -z "$existing_runs" ]; then
        return 1  # No existing runs, so no OOM
    fi
    
    # Check if any run has OOM status
    while IFS= read -r run_dir; do
        if [ -z "$run_dir" ]; then continue; fi
        
        local status=$(analyze_run_status "$run_dir")
        if [[ "$status" == "OOM" ]]; then
            return 0  # Found OOM run
        fi
    done <<< "$existing_runs"
    
    return 1  # No OOM runs found
}

# -----------------------------
# TorchRun Final Logging Flag (for cleaner console)
# -----------------------------
# Quiet C++/glog + c10d
export TORCH_CPP_LOG_LEVEL=ERROR        # hides INFO/WARNING from libtorch/c10d
export GLOG_minloglevel=2               # 0=INFO,1=WARNING,2=ERROR,3=FATAL
export TORCH_DISTRIBUTED_DEBUG=OFF      # don't spam states
export NCCL_DEBUG=ERROR                 # only errors from NCCL
unset TORCH_SHOW_CPP_STACKTRACES        # disable those C++ frames in tracebacks

# -----------------------------
# Debugging Flags
# -----------------------------
# export ENABLE_NSYS=1
# export EXPERIMENT_PYTHON_DEBUG_TRACE_CALLS=1
export EXPERIMENT_LOG_MEMORY_USAGE=0
export EXPERIMENT_SKIP_FA2A_OP=0 # (DebugOnly: Ensure not stuck at fast a2a op)
export EXPERIMENT_SKIP_OPTIMIZER_STEP=1
export EXPERIMENT_NVSHMEM_BUFFER_SIZE_GB=4
export SHOULD_ADD_DEBUG_CASES=0

# ------------------------
# 
# ------------------------
# export NCCL_CUMEM_ENABLE=0
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"
# export PYTORCH_CUDA_ALLOC_CONF="verbose_policy:1"

set -e
export EXPERIMENT_FA2A_SPLIT_SENDRECV=1
export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1
export EXPERIMENT_0TH_SAMPLE_WARMUP_TIMES=1
export EXPERIMENT_WARMUP_TIMES=0
export EXPERIMENT_REPEAT_TIMES=1
# export EXPERIMENT_ADD_SELECTIVE_CKPT=1
# export PYTORCH_CUDA_ALLOC_CONF="pinned_use_background_threads:True"
# export EXPERIMENT_SHOULD_LOG_MEMORY_DURING_WARMUP=1
export MAX_SAMPLE_ID=${MAX_SAMPLE_ID:-5}
export ENABLE_NSYS=0
export EXPERIMENT_ADD_SELECTIVE_CKPT=1
# export EXPERIMENT_ADD_SELECTIVE_CKPT=0

model_config="codellama/CodeLlama-34b-hf 131072 48"
param_configs_cases=()


# -------------------------------
#  Sweep Memory without OOM
# -------------------------------
# memory sweep
config_sweep_memory=()



# Function to check if env_var contains EXPERIMENT__DEBUG=1
has_debug_flag() {
    local env_var="$1"
    
    if [ -z "$env_var" ]; then
        return 1  # No env_var, so no debug flag
    fi
    
    # Remove surrounding quotes if present
    env_var_clean="${env_var//\'/}"  # Remove single quotes
    env_var_clean="${env_var_clean//\"/}"  # Remove double quotes
    
    # Look for EXPERIMENT__DEBUG=1 in the env_var string
    if [[ "$env_var_clean" =~ EXPERIMENT__DEBUG=1 ]]; then
        return 0  # Found debug flag
    fi
    
    return 1  # No debug flag found
}

# Function to check if env_var contains EXPERIMENT__SKIP=1
has_skip_flag() {
    local env_var="$1"
    
    if [ -z "$env_var" ]; then
        return 1  # No env_var, so no skip flag
    fi
    
    # Remove surrounding quotes if present
    env_var_clean="${env_var//\'/}"  # Remove single quotes
    env_var_clean="${env_var_clean//\"/}"  # Remove double quotes
    
    # Look for EXPERIMENT__SKIP=1 in the env_var string
    if [[ "$env_var_clean" =~ EXPERIMENT__SKIP=1 ]]; then
        return 0  # Found skip flag
    fi
    
    return 1  # No skip flag found
}

# Function to extract CHECK_TOTAL_BS from env_var string
# Returns the value if found, empty string if not found
extract_check_total_bs() {
    local env_var="$1"
    
    if [ -z "$env_var" ]; then
        echo ""
        return
    fi
    
    # Remove surrounding quotes if present
    env_var_clean="${env_var//\'/}"  # Remove single quotes
    env_var_clean="${env_var_clean//\"/}"  # Remove double quotes
    
    # Look for CHECK_TOTAL_BS=X in the env_var string
    if [[ "$env_var_clean" =~ CHECK_TOTAL_BS=([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}



# Read config file line by line and append to array
line_number=0
section_env_vars=""  # Store environment variables for current section
in_section=false     # Track if we're inside a section

# Check if Start here marker exists in the file - if not, process all lines
if grep -q "Start here\|$$$ Start here $$$" "$CONFIG_FILE"; then
    found_start_marker=false  # Start here exists, so we need to find it first
    echo "🔍 Start here marker detected in config file, will skip lines until marker is found"
else
    found_start_marker=true   # No Start here marker, process all lines from the beginning
fi

while IFS= read -r line; do
    line_number=$((line_number + 1))
    echo "$line"
    
    # Check for Start here marker: Start processing from this line onwards
    if [[ "$found_start_marker" == "false" ]] && ([[ "$line" == *"Start here"* ]] || [[ "$line" == *"$$$ Start here $$$"* ]]); then
        found_start_marker=true
        echo "🟢 Found Start here marker at line $line_number, starting processing from this line"
        continue
    fi
    
    # Skip lines until we find the Start here marker
    if [[ "$found_start_marker" == "false" ]]; then
        continue
    fi
    
    # Check for Stop here marker: Stop processing when we hit this
    if [[ "$line" == *"Stop here"* ]] || [[ "$line" == *"$$$ Stop here $$$"* ]]; then
        echo "🔴 Found Stop here marker at line $line_number, stopping processing"
        break
    fi
    
    # Check for section start marker: # <<< [env_vars]
    if [[ "$line" =~ ^[[:space:]]*#[[:space:]]*\<\<\<(.*)$ ]]; then
        in_section=true
        section_vars_raw="${BASH_REMATCH[1]}"
        
        # Parse section environment variables and convert to EXPERIMENT__ format
        section_env_vars=""
        if [ ! -z "$section_vars_raw" ]; then
            # Remove leading/trailing whitespace
            section_vars_raw=$(echo "$section_vars_raw" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            # Split by comma and convert each var to EXPERIMENT__ format
            IFS=',' read -ra SECTION_VARS <<< "$section_vars_raw"
            for var in "${SECTION_VARS[@]}"; do
                # Remove whitespace around each variable
                var=$(echo "$var" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                if [ ! -z "$var" ]; then
                    # Convert to EXPERIMENT__ format (e.g., skip=1 -> EXPERIMENT__SKIP=1)
                    if [[ "$var" =~ ^([^=]+)=(.*)$ ]]; then
                        var_name="${BASH_REMATCH[1]}"
                        var_value="${BASH_REMATCH[2]}"
                        experiment_var="${var_name^^}=${var_value}"
                        if [ -z "$section_env_vars" ]; then
                            section_env_vars="$experiment_var"
                        else
                            section_env_vars="$section_env_vars,$experiment_var"
                        fi
                    fi
                fi
            done
        fi
        echo "🔵 Section started with env vars: $section_env_vars"
        continue
    fi
    
    # Check for section end marker: # >>>
    if [[ "$line" =~ ^[[:space:]]*#[[:space:]]*\>\>\>[[:space:]]*$ ]]; then
        in_section=false
        section_env_vars=""
        echo "🔵 Section ended"
        continue
    fi
    
    # Skip empty lines (including whitespace-only) and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*#.*$ ]] && continue
    # Skip the line that contains "Stop here"
    # sleep 1 
    # if there is `#` comment after the line, trim the comment
    original_line="$line"
    if [[ "$line" == *"#"* ]]; then
        line=$(echo "$line" | sed 's/#.*$//')
    fi
    
    # Skip lines that have EXPERIMENT__STATUS=PASS or EXPERIMENT__STATUS=FAIL (filtering logic moved up)
    # Unless we're in LIST_RUNS_FULL mode, which shows all configs including completed ones
    if [[ "$LIST_RUNS_FULL" != "1" ]] && ([[ "$original_line" == *"EXPERIMENT__STATUS=PASS"* ]] || [[ "$original_line" == *"EXPERIMENT__STATUS=FAIL"* ]]); then
        echo "🟢 Skipping config line $line_number due to EXPERIMENT__STATUS flag: $original_line"
        continue
    fi
    
    # Skip lines that have EXPERIMENT__STATUS=PARTIAL when --skip-partial flag is set
    # Unless we're in LIST_RUNS_FULL mode, which shows all configs including completed ones
    if [[ "$SKIP_PARTIAL" == "1" ]] && [[ "$LIST_RUNS_FULL" != "1" ]] && [[ "$original_line" == *"EXPERIMENT__STATUS=PARTIAL"* ]]; then
        echo "🟡 Skipping config line $line_number due to --skip-partial flag (EXPERIMENT__STATUS=PARTIAL): $original_line"
        continue
    fi
    
    # Skip lines that have partial results if --skip-partial flag is set
    if [[ "$SKIP_PARTIAL" == "1" ]] && [[ "$LIST_RUNS_FULL" != "1" ]]; then
        # Parse the config line to check for partial results
        if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$line"; then
            # Create model_path_normalized (assuming codellama/CodeLlama-34b-hf)
            model_path_normalized="codellama_CodeLlama-34b-hf"
            
            if config_has_partial_results "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
                echo "🟡 Skipping config line $line_number due to --skip-partial flag (has benchmark.raw.jsonl): $original_line"
                continue
            fi
        fi
    fi
    
    # Skip lines that have already been started if --skip-started flag is set
    if [[ "$SKIP_STARTED" == "1" ]] && [[ "$LIST_RUNS_FULL" != "1" ]]; then
        # Parse the config line to check for started jobs
        if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$line"; then
            # Create model_path_normalized (assuming codellama/CodeLlama-34b-hf)
            model_path_normalized="codellama_CodeLlama-34b-hf"
            
            if config_has_started "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
                echo "🟠 Skipping config line $line_number due to --skip-started flag (has README.md): $original_line"
                continue
            fi
        fi
    fi
    
    # Skip lines that have been completed with sufficient samples if --skip-completed flag is set
    if [[ "$SKIP_COMPLETED" == "1" ]] && [[ "$LIST_RUNS_FULL" != "1" ]]; then
        # Parse the config line to check for completed jobs
        if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$line"; then
            # Create model_path_normalized (assuming codellama/CodeLlama-34b-hf)
            model_path_normalized="codellama_CodeLlama-34b-hf"
            
            if config_is_completed_with_sufficient_samples "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
                echo "✅ Skipping config line $line_number due to --skip-completed flag (completed with >= $MAX_SAMPLE_ID samples): $original_line"
                continue
            fi
        fi
    fi
    
    # Skip lines that have OOM errors if SKIP_OOM flag is set (default behavior)
    if [[ "$SKIP_OOM" == "1" ]] && [[ "$LIST_RUNS_FULL" != "1" ]]; then
        # Parse the config line to check for OOM jobs
        if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$line"; then
            # Create model_path_normalized (assuming codellama/CodeLlama-34b-hf)
            model_path_normalized="codellama_CodeLlama-34b-hf"
            
            if config_has_oom "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
                echo "🔴 Skipping config line $line_number due to OOM errors (use --no-skip-oom to include): $original_line"
                continue
            fi
        fi
    fi
    
    # Handle debug-only filtering (check both original env_var and section env vars)
    if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$line"; then
        # Combine section env vars with existing env_var for debug checking
        combined_env_var_for_debug="$env_var"
        if [ "$in_section" = true ] && [ ! -z "$section_env_vars" ]; then
            if [ -z "$env_var" ] || [ "$env_var" = "''" ]; then
                combined_env_var_for_debug="'$section_env_vars'"
            else
                env_var_clean="${env_var//\'/}"
                env_var_clean="${env_var_clean//\"/}"
                combined_env_var_for_debug="'$env_var_clean,$section_env_vars'"
            fi
        fi
        
        if [[ "$DEBUG_ONLY" == "1" ]]; then
            # If --debug-only is set, only include configs with EXPERIMENT__DEBUG=1
            if ! has_debug_flag "$combined_env_var_for_debug"; then
                echo "🔍 Skipping config line $line_number (--debug-only set, but no EXPERIMENT__DEBUG=1): $original_line"
                continue
            fi
        else
            # If --debug-only is NOT set, filter OUT configs with EXPERIMENT__DEBUG=1
            if has_debug_flag "$combined_env_var_for_debug"; then
                echo "🔍 Skipping config line $line_number (debug config filtered out, use --debug-only to include): $original_line"
                continue
            fi
        fi
    fi
    
    # Store line number with the config line, appending section env vars if applicable
    modified_line="$line"
    if [ "$in_section" = true ] && [ ! -z "$section_env_vars" ]; then
        # Parse the config line to extract existing env_var
        if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$line"; then
            # Combine section env vars with existing env_var
            if [ -z "$env_var" ] || [ "$env_var" = "''" ]; then
                # No existing env_var, just use section env vars
                combined_env_var="'$section_env_vars'"
            else
                # Remove surrounding quotes from existing env_var
                env_var_clean="${env_var//\'/}"
                env_var_clean="${env_var_clean//\"/}"
                # Combine with section env vars
                combined_env_var="'$env_var_clean,$section_env_vars'"
            fi
            
            # Reconstruct the line with the combined env_var
            modified_line="$nnodes $batch_size $microbatch_size $num_tokens $mode $cp_size $pp_size $tp_size $comment $combined_env_var"
            echo "🔵 Applied section env vars to line $line_number: $section_env_vars"
        fi
    fi
    
    # Check for EXPERIMENT__SKIP=1 flag in the final modified line
    if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$modified_line"; then
        if has_skip_flag "$env_var"; then
            echo "⏭️  Skipping config line $line_number due to EXPERIMENT__SKIP=1: $original_line"
            continue
        fi
    fi
    
    # Check if CHECK_TOTAL_BS is specified in env_var and verify bs * mb = CHECK_TOTAL_BS
    if read -r nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$modified_line"; then
        check_total_bs=$(extract_check_total_bs "$env_var")
        if [ ! -z "$check_total_bs" ]; then
            total_bs=$((batch_size * microbatch_size))
            if [ "$total_bs" -ne "$check_total_bs" ]; then
                echo "📊 Skipping config line $line_number due to CHECK_TOTAL_BS=$check_total_bs (bs($batch_size) * mb($microbatch_size) = $total_bs): $original_line"
                continue
            fi
        fi
    fi
    
    config_sweep_memory+=("$line_number:$modified_line")
done < "$CONFIG_FILE"

for config_with_line in "${config_sweep_memory[@]}"; do
    # Extract line number and config
    line_num="${config_with_line%%:*}"
    config="${config_with_line#*:}"
    cases+=("$line_num $model_config $config")
done

# Function to extract EXPERIMENT_SCHEDULE_PRIORITY from env_var string
extract_schedule_priority() {
    local env_var="$1"
    local priority=10
    
    if [ ! -z "$env_var" ]; then
        # Remove surrounding quotes if present
        env_var_clean="${env_var//\'/}"  # Remove single quotes
        env_var_clean="${env_var_clean//\"/}"  # Remove double quotes
        
        # Look for EXPERIMENT_SCHEDULE_PRIORITY=X in the env_var string
        if [[ "$env_var_clean" =~ EXPERIMENT_SCHEDULE_PRIORITY=([0-9]+) ]]; then
            priority="${BASH_REMATCH[1]}"
        fi
    fi
    
    echo "$priority"
}

# Function to sort cases by EXPERIMENT_SCHEDULE_PRIORITY
sort_cases_by_priority() {
    local temp_file=$(mktemp)
    local -a new_cases=()
    
    # Write priority:case pairs to temporary file
    for case in "${cases[@]}"; do
        # Extract env_var using regex - it's the part between the last two single quotes
        local env_var=""
        if [[ "$case" =~ \'([^\']+)\'[[:space:]]*$ ]]; then
            env_var="${BASH_REMATCH[1]}"
        fi
        
        priority=$(extract_schedule_priority "$env_var")
        printf "%02d:%s\n" "$priority" "$case" >> "$temp_file"
    done
    
    # Sort by priority (descending order - higher priority first) and rebuild array
    while IFS=':' read -r priority case_line; do
        # Skip the priority part and get the rest of the line
        full_case="${case_line}"
        new_cases+=("$full_case")
    done < <(sort -t: -k1,1nr "$temp_file")
    
    # Replace the original cases array
    cases=("${new_cases[@]}")
    
    # Clean up
    rm -f "$temp_file"
}

# Sort cases by EXPERIMENT_SCHEDULE_PRIORITY (higher priority first)
sort_cases_by_priority

# for config in "${param_configs_cases[@]}"; do
#     cases+=("$model_config $config")
# done


function write_status_log() {
    echo "$1" >> $STATUS_FILE_PATH
}

# Function to monitor job liveness and detect stuck conditions
monitor_job_liveness() {
    local job_pid=$1
    local run_output_dir=$2
    local liveness_file="$run_output_dir/job_liveness.txt"
    local stuck_warning_file="$run_output_dir/stuck_warnings.txt"
    
    echo "🟡 Starting job liveness monitor for PID $job_pid in $run_output_dir"
    
    # Create the run output directory if it doesn't exist
    mkdir -p "$run_output_dir"
    
        # Initialize tracking variables (in-memory to avoid file I/O)
    local last_liveness_write=0
    local last_warning_time=0
    
    while kill -0 "$job_pid" 2>/dev/null; do
        current_time=$(date +%s)
        
        # Write liveness timestamp less frequently (every 30 seconds instead of 10)
        # This reduces NFS writes by 67%
        if [ $((current_time - last_liveness_write)) -ge 30 ]; then
            echo "$current_time" > "$liveness_file"
            last_liveness_write=$current_time
        fi
        
        # Check for stuck condition (job alive but logs not updating)
        if [ -d "$run_output_dir/logs" ]; then
            # Use directory timestamp instead of scanning all files (MUCH faster on NFS!)
            # Directory mtime updates when any file inside is created/modified
            logs_dir_time=$(stat -c %Y "$run_output_dir/logs" 2>/dev/null || echo "0")
            
            if [ "$logs_dir_time" != "0" ]; then
                log_age_seconds=$((current_time - logs_dir_time))
                log_age_minutes=$((log_age_seconds / 60))
                
                # Check if logs directory hasn't been updated for >5 minutes while job is alive
                if [ $log_age_minutes -gt 3 ]; then
                    # Print warning every 20 seconds (using in-memory variable instead of file I/O)
                    if [ $((current_time - last_warning_time)) -ge 20 ]; then
                        echo "🟡 MAYBE_STUCK detected: Job PID $job_pid alive but logs dir not updated for $log_age_minutes minutes"
                        last_warning_time=$current_time
                        # Only write to file when warning occurs to reduce NFS I/O
                        echo "$current_time" > "$stuck_warning_file"
                    fi
                fi
            fi
        fi
        
        # Check every 10 seconds
        sleep 45
    done
    
    echo "🟡 Job liveness monitor finished for PID $job_pid"
}


# -------------------------------
# -------------------------------

wlbllm_cases=()
wlbllm_perseq_cases=()
d2_cases=()
# Define format string once
format_str="%-4s  %-15s  %5s  %6s  %4s  %4s  %4s  %6s  %7s  %4s  %4s  %4s  %4s %10s\n"

# Display active filters
filters_active=false
if [ ! -z "$FILTER_MODEL_PATH" ] || [ ! -z "$FILTER_ATB" ] || [ ! -z "$FILTER_NLAYERS" ] || [ ! -z "$FILTER_N" ] || [ ! -z "$FILTER_BS" ] || [ ! -z "$FILTER_MB" ] || [ ! -z "$FILTER_TOKS" ] || [ ! -z "$FILTER_MODE" ] || [ ! -z "$FILTER_CP" ] || [ ! -z "$FILTER_PP" ] || [ ! -z "$FILTER_TP" ] || [ ! -z "$FILTER_PRIORITY_GE" ] || [ ! -z "$FILTER_PRIORITY_LE" ]; then
    filters_active=true
    echo ""
    echo "🔍 Active Filters:"
    [ ! -z "$FILTER_MODEL_PATH" ] && echo "  model_path: $FILTER_MODEL_PATH"
    [ ! -z "$FILTER_ATB" ] && echo "  atb: $FILTER_ATB"
    [ ! -z "$FILTER_NLAYERS" ] && echo "  nlayers: $FILTER_NLAYERS"
    [ ! -z "$FILTER_N" ] && echo "  N (nodes): $FILTER_N"
    [ ! -z "$FILTER_BS" ] && echo "  bs (batch_size): $FILTER_BS"
    [ ! -z "$FILTER_MB" ] && echo "  mb (microbatch): $FILTER_MB"
    [ ! -z "$FILTER_TOKS" ] && echo "  toks (tokens): $FILTER_TOKS"
    [ ! -z "$FILTER_MODE" ] && echo "  mode: $FILTER_MODE"
    [ ! -z "$FILTER_CP" ] && echo "  cp: $FILTER_CP"
    [ ! -z "$FILTER_PP" ] && echo "  pp: $FILTER_PP"
    [ ! -z "$FILTER_TP" ] && echo "  tp: $FILTER_TP"
    [ ! -z "$FILTER_PRIORITY_GE" ] && echo "  priority >= $FILTER_PRIORITY_GE"
    [ ! -z "$FILTER_PRIORITY_LE" ] && echo "  priority <= $FILTER_PRIORITY_LE"
    echo ""
fi

FAILED_SANITY_CHECK=0

if [ "$filters_active" = true ]; then
    echo "🟡 Filtered cases:"
else
    echo "🟡 All cases:"
fi
printf "$format_str" \
    "L#" "model_path" "atb" "nlayers" "N" "bs" "mb" "toks" "mode" "cp" "pp" "tp" "comment" "env_var"
for case in "${cases[@]}"; do
    read -r line_num model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$case"
    
    # Apply filters - skip this case if it doesn't match
    if ! case_matches_filters "$model_path" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$env_var"; then
        continue
    fi
    
    # Sanity check: For d2 mode, cp * pp must equal N (nodes)
    if [ "$mode" = "d2" ]; then
        expected_nodes=$((cp_size * pp_size))
        if [ "$expected_nodes" -ne "$nnodes" ]; then
            echo "❌ SANITY CHECK FAILED for line $line_num: mode=d2 requires cp * pp == N, but cp($cp_size) * pp($pp_size) = $expected_nodes != N($nnodes)"
            FAILED_SANITY_CHECK=1
            continue
        fi
    fi
    # Trim model path to 15 chars
    model_path_short="${model_path:0:15}"
    
    # Create model_path_normalized for folder search
    model_path_normalized=$(echo $model_path | sed 's/\//_/g')
    
    if [ "$mode" == "wlbllm" ]; then
        wlbllm_cases+=("$case")
        
        # Check if this config would be skipped due to partial results, started status, completion, or OOM
        skip_indicator=""
        if [[ "$SKIP_PARTIAL" == "1" ]] && config_has_partial_results "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-PARTIAL]"
        elif [[ "$SKIP_STARTED" == "1" ]] && config_has_started "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-STARTED]"
        elif [[ "$SKIP_COMPLETED" == "1" ]] && config_is_completed_with_sufficient_samples "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-COMPLETED]"
        elif [[ "$SKIP_OOM" == "1" ]] && config_has_oom "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-OOM]"
        fi
        
        printf "$format_str" \
            "$line_num" "$model_path_short" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$comment" "${env_var}${skip_indicator}"
        
        # Show existing runs if --ls-runs or --ls-runs-full is specified
        if [ "$LIST_RUNS" == "1" ] || [ "$LIST_RUNS_FULL" == "1" ]; then
            existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
            if [ ! -z "$existing_runs" ]; then
                echo "    📁 Existing runs:"
                while IFS= read -r run_dir; do
                    run_name=$(basename "$run_dir")
                    status=$(analyze_run_status "$run_dir")
                    status_display=$(get_status_display "$status")
                    echo "       $run_name - $status_display"
                done <<< "$existing_runs"
            else
                echo "    📁 No existing runs found"
            fi
            echo ""
        fi
    fi
    if [ "$mode" == "wlbllm_perseq" ]; then
        wlbllm_perseq_cases+=("$case")
        
        # Check if this config would be skipped due to partial results, started status, completion, or OOM
        skip_indicator=""
        if [[ "$SKIP_PARTIAL" == "1" ]] && config_has_partial_results "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-PARTIAL]"
        elif [[ "$SKIP_STARTED" == "1" ]] && config_has_started "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-STARTED]"
        elif [[ "$SKIP_COMPLETED" == "1" ]] && config_is_completed_with_sufficient_samples "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-COMPLETED]"
        elif [[ "$SKIP_OOM" == "1" ]] && config_has_oom "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-OOM]"
        fi
        
        printf "$format_str" \
            "$line_num" "$model_path_short" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$comment" "${env_var}${skip_indicator}"
        
        # Show existing runs if --ls-runs or --ls-runs-full is specified
        if [ "$LIST_RUNS" == "1" ] || [ "$LIST_RUNS_FULL" == "1" ]; then
            existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
            if [ ! -z "$existing_runs" ]; then
                echo "    📁 Existing runs:"
                while IFS= read -r run_dir; do
                    run_name=$(basename "$run_dir")
                    status=$(analyze_run_status "$run_dir")
                    status_display=$(get_status_display "$status")
                    echo "       $run_name - $status_display"
                done <<< "$existing_runs"
            else
                echo "    📁 No existing runs found"
            fi
            echo ""
        fi
    fi
    if [ "$mode" == "d2" ]; then
        d2_cases+=("$case")
        
        # Check if this config would be skipped due to partial results, started status, completion, or OOM
        skip_indicator=""
        if [[ "$SKIP_PARTIAL" == "1" ]] && config_has_partial_results "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-PARTIAL]"
        elif [[ "$SKIP_STARTED" == "1" ]] && config_has_started "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-STARTED]"
        elif [[ "$SKIP_COMPLETED" == "1" ]] && config_is_completed_with_sufficient_samples "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-COMPLETED]"
        elif [[ "$SKIP_OOM" == "1" ]] && config_has_oom "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-OOM]"
        fi
        
        printf "$format_str" \
            "$line_num" "$model_path_short" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$comment" "${env_var}${skip_indicator}"
        
        # Show existing runs if --ls-runs or --ls-runs-full is specified
        if [ "$LIST_RUNS" == "1" ] || [ "$LIST_RUNS_FULL" == "1" ]; then
            existing_runs=$(find_existing_runs "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized")
            if [ ! -z "$existing_runs" ]; then
                echo "    📁 Existing runs:"
                while IFS= read -r run_dir; do
                    run_name=$(basename "$run_dir")
                    status=$(analyze_run_status "$run_dir")
                    status_display=$(get_status_display "$status")
                    echo "       $run_name - $status_display"
                done <<< "$existing_runs"
            else
                echo "    📁 No existing runs found"
            fi
            echo ""
        fi
    fi
done
echo "🟡 Total wlbllm cases: ${#wlbllm_cases[@]}"
echo "🟡 Total wlbllm_perseq cases: ${#wlbllm_perseq_cases[@]}"
echo "🟡 Total d2 cases: ${#d2_cases[@]}"

# Show skip information if flags are enabled
if [[ "$SKIP_PARTIAL" == "1" ]]; then
    echo "🟡 --skip-partial flag is enabled"
fi
if [[ "$SKIP_STARTED" == "1" ]]; then
    echo "🟠 --skip-started flag is enabled"
fi
if [[ "$SKIP_COMPLETED" == "1" ]]; then
    echo "✅ --skip-completed flag is enabled (MAX_SAMPLE_ID: $MAX_SAMPLE_ID)"
fi
if [[ "$SKIP_OOM" == "1" ]]; then
    echo "🔴 Skipping OOM cases by default (use --no-skip-oom to include them)"
else
    echo "🔴 --no-skip-oom flag is enabled (including OOM cases)"
fi
if [[ "$MONITOR_STUCK" == "1" ]]; then
    echo "🔍 --monitor-stuck flag is enabled (job monitoring active)"
else
    echo "🏃 Job monitoring is disabled (use --monitor-stuck to enable)"
fi
if [[ "$DEBUG_ONLY" == "1" ]]; then
    echo "🔍 --debug-only flag is enabled (only running configs with EXPERIMENT__DEBUG=1)"
else
    echo "🔍 Debug configs are filtered out by default (use --debug-only to include only debug configs)"
fi

echo ""

# Exit if only listing configs
if [ "$LIST_ONLY" == "1" ]; then
    echo "📋 LIST_ONLY mode: Showing configs that would be run and exiting."
    exit 0
fi

# Exit if only listing configs with existing runs
if [ "$LIST_RUNS" == "1" ]; then
    echo "📋 LIST_RUNS mode: Showing configs with existing run folders and exiting."
    exit 0
fi

# Exit if listing all configs with existing runs (including completed)
if [ "$LIST_RUNS_FULL" == "1" ]; then
    echo "📋 LIST_RUNS_FULL mode: Showing ALL configs (including completed) with existing run folders and exiting."
    exit 0
fi

if [ "$FAILED_SANITY_CHECK" == "1" ]; then
    echo "❌ SANITY CHECK FAILED. Exiting."
    exit 1
fi

sleep 3



max_cases=10000
echo "🏁 Start regression sweep. Only running $max_cases cases."
cases_index=0

set -x


for sample_config in "${dists[@]}"; do
for config in "${cases[@]}"; do
    echo config="$config" ";" sample_config="$sample_config" ";"
    if ! read -r line_num model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$config"; then
        echo "⚠️ Failed to parse config: $config"
        continue
    fi
    
    # Apply filters - skip this case if it doesn't match
    if ! case_matches_filters "$model_path" "$attn_linear_breakpoint" "$num_layers" "$nnodes" "$batch_size" "$microbatch_size" "$num_tokens" "$mode" "$cp_size" "$pp_size" "$tp_size" "$env_var"; then
        echo "🔍 Skipping config due to filters: $config"
        continue
    fi
    
    # Sanity check: For d2 mode, cp * pp must equal N (nodes)
    if [ "$mode" = "d2" ]; then
        expected_nodes=$((cp_size * pp_size))
        if [ "$expected_nodes" -ne "$nnodes" ]; then
            echo "❌ SANITY CHECK FAILED for line $line_num: mode=d2 requires cp * pp == N, but cp($cp_size) * pp($pp_size) = $expected_nodes != N($nnodes)"
            continue
        fi
    fi
    if ! read -r sample_name change_long_doc_ratio <<< "$sample_config"; then
        echo "⚠️ Failed to parse sample config: $sample_config" 
        continue
    fi
    
    export MODE=$mode
    export BATCH_SIZE=$batch_size
    export NNODES=$nnodes
    export NUM_TOKENS=$num_tokens
    export NUM_MICROBATCH=$microbatch_size
    export CP_SIZE=$cp_size
    export PP_SIZE=$pp_size
    export TP_SIZE=$tp_size
    export ATTN_LINEAR_BREAKPOINT=$attn_linear_breakpoint
    export NUM_LAYERS=$num_layers
    export MODEL_PATH=$model_path
    export MODEL_PATH_normalized=$(echo $model_path | sed 's/\//_/g')
    export SAMPLE_NAME=$sample_name
    export CHANGE_LONG_DOC_RATIO=$change_long_doc_ratio

    export RATIO=$(echo "8 / ($NNODES / $BATCH_SIZE)" | bc -l)
    # export OUTPUT_DIR_SUFFIX_ADDON="${OUTPUT_DIR_SUFFIX_ADDON}"


    export ELONGATE_FACTOR=$(($NUM_TOKENS / 65536))
    if [ $ELONGATE_FACTOR -lt 1 ]; then
        ELONGATE_FACTOR=1
    fi
    
    
    # ----------------------------------
    #  Memory Sweep Pass
    # ----------------------------------
    # Override export environment variables if env_var is not empty
    declare -A old_env_vars  # Associative array to store old values
    env_vars_to_restore=()  # Array to track which variables we modified
    
    if [ ! -z "$env_var" ]; then
        # Remove surrounding quotes if present
        env_var_clean="${env_var//\'/}"  # Remove single quotes
        env_var_clean="${env_var_clean//\"/}"  # Remove double quotes
        
        IFS=',' read -ra ENV_VARS <<< "$env_var_clean"
        for var in "${ENV_VARS[@]}"; do
            IFS='=' read -r key value <<< "$var"
            
            # Save the old value if the variable exists
            if [[ -v "$key" ]]; then
                old_env_vars["$key"]="${!key}"
                echo "⚪ Saving old environment variable: $key=${!key}"
            else
                old_env_vars["$key"]="__UNSET__"
                # echo "⚪ Variable $key was not previously set"
            fi
            env_vars_to_restore+=("$key")
            
            # set -x
            export "$key"="$value"
            # set +x
            echo "🟡 Exporting environment variable: $key=$value"
        done
    fi
    
    
    

    # D2 normal
    echo "🟡 Running config (line $line_num): MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT, RATIO=$RATIO"

    if [ "$DRY_RUN" == "1" ]; then
        echo "🔵 DRY_RUN=1, skipping execution"
        continue
    fi

    

    write_status_log "Running: $sample_config $config"
    
    # Construct the expected output directory for this run
    run_timestamp="${TS}"
    
    # Check if FOLDER_SEPARATOR flag is enabled for simplified directory naming
    if [ "${FOLDER_SEPARATOR:-0}" == "1" ]; then
        run_output_dir="${OUTPUT_DIR_PREFIX}/${run_timestamp}.-----"
        echo "🔧 FOLDER_SEPARATOR=1: Using simplified directory name: ${run_timestamp}.-----"
    else
        run_output_dir="${OUTPUT_DIR_PREFIX}/${run_timestamp}.${MODE}-n${NNODES}-t${NUM_TOKENS}-b${BATCH_SIZE}-mb${NUM_MICROBATCH}-cp${CP_SIZE}tp${TP_SIZE}pp${PP_SIZE}${OUTPUT_DIR_SUFFIX_ADDON}"
    fi

    # echo CHANGE_LONG_DOC_RATIO=$CHANGE_LONG_DOC_RATIO
    # exit 0
    
    if [ $MODE == "d2" ]; then
        if [ "$MONITOR_STUCK" == "1" ]; then
            echo "🔍 Running with monitoring enabled"
            # Start job in background and get PID
            bash test_megatron_e2e_pipeline_with_cp.sh &
            job_pid=$!
            
            # Start monitoring in background
            monitor_job_liveness "$job_pid" "$run_output_dir" &
            monitor_pid=$!
            
            # Wait for the main job to complete
            wait "$job_pid"
            job_exit_code=$?
            
            # Kill the monitor
            kill "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
        else
            echo "🏃 Running without monitoring (use --monitor-stuck to enable)"
            # Run job normally without monitoring
            bash test_megatron_e2e_pipeline_with_cp.sh
            job_exit_code=$?
        fi
    fi

    if [ $MODE == "wlbllm" ]; then
        if [ "$MONITOR_STUCK" == "1" ]; then
            echo "🔍 Running with monitoring enabled"
            # Start job in background and get PID
            bash test_wlb_e2e.sh &
            job_pid=$!
            
            # Start monitoring in background
            monitor_job_liveness "$job_pid" "$run_output_dir" &
            monitor_pid=$!
            
            # Wait for the main job to complete
            wait "$job_pid"
            job_exit_code=$?
            
            # Kill the monitor
            kill "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
        else
            echo "🏃 Running without monitoring (use --monitor-stuck to enable)"
            # Run job normally without monitoring
            bash test_wlb_e2e.sh
            job_exit_code=$?
        fi
    fi

    if [ $MODE == "wlbllm_perseq" ]; then
        if [ "$MONITOR_STUCK" == "1" ]; then
            echo "🔍 Running with monitoring enabled"
            # Start job in background and get PID
            bash test_wlb_e2e.sh &
            job_pid=$!
            
            # Start monitoring in background
            monitor_job_liveness "$job_pid" "$run_output_dir" &
            monitor_pid=$!
            
            # Wait for the main job to complete
            wait "$job_pid"
            job_exit_code=$?
            
            # Kill the monitor
            kill "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
        else
            echo "🏃 Running without monitoring (use --monitor-stuck to enable)"
            # Run job normally without monitoring
            bash test_wlb_e2e.sh
            job_exit_code=$?
        fi
    fi

    write_status_log "Finished: $sample_config $config"

    echo "🟡 Finished running config (line $line_num): MODE=$MODE, NNODES=$NNODES, BATCH_SIZE=$BATCH_SIZE, NUM_TOKENS=$NUM_TOKENS, NUM_MICROBATCH=$NUM_MICROBATCH, CP_SIZE=$CP_SIZE, PP_SIZE=$PP_SIZE, TP_SIZE=$TP_SIZE, MODE=$MODE, MODEL_PATH=$MODEL_PATH, MODEL_PATH_normalized=$MODEL_PATH_normalized, NUM_LAYERS=$NUM_LAYERS, ATTN_LINEAR_BREAKPOINT=$ATTN_LINEAR_BREAKPOINT, COMMENT=$COMMENT, RATIO=$RATIO"
    
    # Restore original environment variables
    if [ ${#env_vars_to_restore[@]} -gt 0 ]; then
        # echo "⚪ Restoring original environment variables..."
        for key in "${env_vars_to_restore[@]}"; do
            if [ "${old_env_vars[$key]}" == "__UNSET__" ]; then
                unset "$key"
                # echo "⚪ Unsetting environment variable: $key (was not previously set)"
            else
                export "$key"="${old_env_vars[$key]}"
                # echo "⚪ Restored environment variable: $key=${old_env_vars[$key]}"
            fi
        done
    fi


    # ----------------------------------
    #  Performance Sweep Pass
    # ----------------------------------
    # TODO: Add logic to check if the performance run is completed. 
    # if [ "${EXPERIMENT__STATUS:-}" == "FAIL" ]; then
    #     echo "🔴 Skipping performance sweep due to EXPERIMENT__STATUS=FAIL flag"
    #     continue
    # fi

    # OLD_ENABLE_NSYS=$ENABLE_NSYS
    # OLD_MAX_SAMPLE_ID=$MAX_SAMPLE_ID
    # export MAX_SAMPLE_ID=10
    # export ENABLE_NSYS=0

    # if [ $MODE == "d2" ]; then
    #     bash test_megatron_e2e_pipeline_with_cp.sh
    # fi

    # if [ $MODE == "wlbllm" ]; then
    #     bash test_wlb_e2e.sh
    # fi

    # export MAX_SAMPLE_ID=$OLD_MAX_SAMPLE_ID
    # export ENABLE_NSYS=$OLD_ENABLE_NSYS
    

    # ----------------------------------
    #  Epilogue
    # ----------------------------------
    
    cases_index=$((cases_index + 1))
    if [ $cases_index -gt $max_cases ]; then
        break
    fi

done
done

write_status_log "Finished running all cases"

for i in {1..10}; do
    echo '\a'
    sleep 1
done