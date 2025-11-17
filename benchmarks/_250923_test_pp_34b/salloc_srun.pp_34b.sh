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
MONITOR_STUCK=0

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
        --monitor-stuck)
            MONITOR_STUCK=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--ls] [--ls-runs] [--ls-runs-full] [--dry-run] [--skip-partial] [--skip-started] [--monitor-stuck]"
            echo "  --ls            List configs that would be run and exit"
            echo "  --ls-runs       List configs and show existing run folders, then exit"
            echo "  --ls-runs-full  List ALL configs (including completed) and show existing run folders, then exit"
            echo "  --dry-run       Show what would be run but don't execute"
            echo "  --skip-partial  Skip configs that already have partial results (benchmark.raw.jsonl)"
            echo "  --skip-started  Skip configs that have already been started (have README.md)"
            echo "  --monitor-stuck Enable job monitoring to detect stuck processes (disabled by default)"
            exit 1
            ;;
    esac
done

# export JOBID=${JOBID:-710588}
TS=$(TZ=America/Los_Angeles date +%m%d_%H%M%S)_PST
export OUTPUT_DIR_PREFIX=/mnt/weka/home/yonghao.zhuang/jd/d2/benchmarks/_250923_test_pp_34b/logs.v1-sweep-pp-34b

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
    local search_pattern="*.${mode}-n${nnodes}-t${num_tokens}-b${batch_size}-mb${microbatch_size}-cp${cp_size}tp${tp_size}pp${pp_size}-${model_path_normalized}"
    
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
export MAX_SAMPLE_ID=1
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
export MAX_SAMPLE_ID=1
export ENABLE_NSYS=1
# Get current directory of this script
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read config file line by line and append to array
line_number=0
while IFS= read -r line; do
    line_number=$((line_number + 1))
    echo "$line"
    if [[ "$line" == *"Stop here"* ]]; then
        break
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
    
    # Store line number with the config line
    config_sweep_memory+=("$line_number:$line")
done < "$CURDIR/config_sweep_memory.config.sh"

for config_with_line in "${config_sweep_memory[@]}"; do
    # Extract line number and config
    line_num="${config_with_line%%:*}"
    config="${config_with_line#*:}"
    cases+=("$line_num $model_config $config")
done


# for config in "${param_configs_cases[@]}"; do
#     cases+=("$model_config $config")
# done


dists=(
    "wlbllm 0.0"
    # "prolong 0.3"
)

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
d2_cases=()
# Define format string once
format_str="%-4s  %-15s  %5s  %6s  %4s  %4s  %4s  %6s  %7s  %4s  %4s  %4s  %4s %10s\n"

echo "🟡 All cases:"
printf "$format_str" \
    "L#" "model_path" "atb" "nlayers" "N" "bs" "mb" "toks" "mode" "cp" "pp" "tp" "comment" "env_var"
for case in "${cases[@]}"; do
    read -r line_num model_path attn_linear_breakpoint num_layers nnodes batch_size microbatch_size num_tokens mode cp_size pp_size tp_size comment env_var <<< "$case"
    # Trim model path to 15 chars
    model_path_short="${model_path:0:15}"
    
    # Create model_path_normalized for folder search
    model_path_normalized=$(echo $model_path | sed 's/\//_/g')
    
    if [ "$mode" == "wlbllm" ]; then
        wlbllm_cases+=("$case")
        
        # Check if this config would be skipped due to partial results or started status
        skip_indicator=""
        if [[ "$SKIP_PARTIAL" == "1" ]] && config_has_partial_results "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-PARTIAL]"
        elif [[ "$SKIP_STARTED" == "1" ]] && config_has_started "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-STARTED]"
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
        
        # Check if this config would be skipped due to partial results or started status
        skip_indicator=""
        if [[ "$SKIP_PARTIAL" == "1" ]] && config_has_partial_results "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-PARTIAL]"
        elif [[ "$SKIP_STARTED" == "1" ]] && config_has_started "$mode" "$nnodes" "$num_tokens" "$batch_size" "$microbatch_size" "$cp_size" "$tp_size" "$pp_size" "$model_path_normalized"; then
            skip_indicator=" [SKIP-STARTED]"
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
echo "🟡 Total d2 cases: ${#d2_cases[@]}"

# Show skip information if flags are enabled
if [[ "$SKIP_PARTIAL" == "1" ]]; then
    echo "🟡 --skip-partial flag is enabled"
fi
if [[ "$SKIP_STARTED" == "1" ]]; then
    echo "🟠 --skip-started flag is enabled"
fi
if [[ "$MONITOR_STUCK" == "1" ]]; then
    echo "🔍 --monitor-stuck flag is enabled (job monitoring active)"
else
    echo "🏃 Job monitoring is disabled (use --monitor-stuck to enable)"
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
    export OUTPUT_DIR_SUFFIX_ADDON="-${MODEL_PATH_normalized}"


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