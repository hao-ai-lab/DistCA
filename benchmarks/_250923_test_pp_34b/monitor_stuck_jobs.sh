#!/bin/bash

# Standalone job stuck monitor
# Monitors the latest job in the output directory for stuck conditions

set -e

# Configuration
CHECK_INTERVAL_SECONDS=15
STUCK_THRESHOLD_MINUTES=3

# Command line argument parsing
OUTPUT_DIR_PREFIX=""
VERBOSE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 <OUTPUT_DIR_PREFIX> [--verbose]"
            echo ""
            echo "Monitor the latest job for stuck conditions"
            echo ""
            echo "Arguments:"
            echo "  OUTPUT_DIR_PREFIX  Directory containing job folders (required)"
            echo "  --verbose          Enable verbose logging"
            echo ""
            echo "Example:"
            echo "  $0 /path/to/logs"
            echo "  $0 /path/to/logs --verbose"
            echo ""
            echo "The script will:"
            echo "  - Find the lexicographically latest job folder"
            echo "  - Monitor the most recent log file timestamp for stuck detection"
            echo "  - Check every $CHECK_INTERVAL_SECONDS seconds"
            echo "  - Alert when latest log file not updated for >$STUCK_THRESHOLD_MINUTES minutes"
            exit 0
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        *)
            # Assume it's the output directory if it starts with / or doesn't start with --
            if [[ "$1" == /* ]] || [[ "$1" != --* ]]; then
                OUTPUT_DIR_PREFIX="$1"
            else
                echo "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$OUTPUT_DIR_PREFIX" ]; then
    echo "Error: OUTPUT_DIR_PREFIX is required"
    echo "Use --help for usage information"
    exit 1
fi

# Function to log with timestamp
log() {
    local level="$1"
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $*"
}

# Function to find the latest job folder
find_latest_job_folder() {
    local output_dir="$1"
    
    if [ ! -d "$output_dir" ]; then
        log "ERROR" "Output directory does not exist: $output_dir"
        return 1
    fi
    
    # Find the lexicographically latest folder (most recent timestamp)
    local latest_folder=$(find "$output_dir" -maxdepth 1 -type d -name "*.*-n*-t*-b*-mb*-cp*tp*pp*-*" 2>/dev/null | sort | tail -1)
    
    if [ -z "$latest_folder" ]; then
        log "WARNING" "No job folders found in $output_dir"
        return 1
    fi
    
    echo "$latest_folder"
    return 0
}

# Function to analyze benchmark file and return info
analyze_benchmark_file() {
    local job_dir="$1"
    local benchmark_file="$job_dir/benchmark.raw.jsonl"
    
    if [ ! -f "$benchmark_file" ]; then
        echo ""
        return
    fi
    
    # Count rows in the file (excluding empty lines)
    local row_count=$(grep -c '.' "$benchmark_file" 2>/dev/null || echo "0")
    
    # Calculate average duration from duration_ms field
    local avg_duration="N/A"
    if [ "$row_count" -gt 0 ]; then
        # Extract duration_ms values and calculate average
        local total_duration=0
        local valid_durations=0
        
        while IFS= read -r line; do
            # Skip empty lines
            if [ -n "$line" ]; then
                # Extract duration_ms using grep and sed (more portable than jq)
                local duration=$(echo "$line" | sed -n 's/.*"duration_ms": *\([0-9.]*\).*/\1/p')
                if [ -n "$duration" ]; then
                    # Use bc for floating point arithmetic if available, otherwise use awk
                    if command -v bc >/dev/null 2>&1; then
                        total_duration=$(echo "$total_duration + $duration" | bc)
                    else
                        total_duration=$(awk "BEGIN {print $total_duration + $duration}")
                    fi
                    valid_durations=$((valid_durations + 1))
                fi
            fi
        done < "$benchmark_file"
        
        if [ "$valid_durations" -gt 0 ]; then
            if command -v bc >/dev/null 2>&1; then
                avg_duration=$(echo "scale=2; $total_duration / $valid_durations" | bc)
            else
                avg_duration=$(awk "BEGIN {printf \"%.2f\", $total_duration / $valid_durations}")
            fi
            avg_duration="${avg_duration}ms"
        fi
    fi
    
    # Read all lines for preview (only non-empty)
    local preview=""
    if [ -r "$benchmark_file" ] && [ "$row_count" -gt 0 ]; then
        preview=$(grep -v '^$' "$benchmark_file" 2>/dev/null | tr '\n' '|' | sed 's/|$//' | cut -c1-130)
    fi
    
    echo "${row_count}:${avg_duration}:${preview}"
}

# Function to analyze job status for stuck detection
analyze_job_for_stuck() {
    local job_dir="$1"
    local current_time=$(date +%s)
    
    # Check if job started (has README.md)
    if [ ! -f "$job_dir/README.md" ]; then
        echo "NOT_STARTED"
        return
    fi
    
    # Check if job completed (has benchmark.json or benchmark.raw.jsonl)
    if [ -f "$job_dir/benchmark.json" ]; then
        echo "COMPLETED:benchmark.json"
        return
    elif [ -f "$job_dir/benchmark.raw.jsonl" ]; then
        local benchmark_info=$(analyze_benchmark_file "$job_dir")
        echo "COMPLETED:benchmark.raw.jsonl:${benchmark_info}"
        return
    fi
    
    # Check log files update times - this is our primary stuck indicator
    if [ -d "$job_dir/logs" ]; then
        # Find the most recently modified log file
        local latest_log_file=$(find "$job_dir/logs" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -n "$latest_log_file" ] && [ -f "$latest_log_file" ]; then
            # Get the timestamp of the most recent log file
            local latest_log_time=$(stat -c %Y "$latest_log_file" 2>/dev/null || echo "0")
            
            if [ "$latest_log_time" != "0" ]; then
                local log_age_seconds=$((current_time - latest_log_time))
                local log_age_minutes=$((log_age_seconds / 60))
                
                # Check if the latest log file hasn't been updated for >threshold minutes
                if [ $log_age_minutes -gt $STUCK_THRESHOLD_MINUTES ]; then
                    echo "STUCK:${log_age_minutes}min"
                    return
                else
                    # Job is actively updating logs
                    echo "RUNNING:${log_age_minutes}min"
                    return
                fi
            else
                # Log file exists but no timestamp - unusual case
                echo "RUNNING:NO_TIMESTAMP"
                return
            fi
        else
            # Logs directory exists but no log files yet
            echo "STARTING:NO_LOGS"
            return
        fi
    else
        # Job started but no logs directory yet - probably just starting
        echo "STARTING"
        return
    fi
}

# Function to get status display
get_status_display() {
    local status_info="$1"
    
    case $status_info in
        "NOT_STARTED")
            echo "⚪ Job not started yet"
            ;;
        "COMPLETED:benchmark.json")
            echo "✅ Job completed (benchmark.json found)"
            ;;
        "COMPLETED:benchmark.raw.jsonl:"*)
            local benchmark_details="${status_info#COMPLETED:benchmark.raw.jsonl:}"
            if [ -n "$benchmark_details" ]; then
                local row_count=$(echo "$benchmark_details" | cut -d':' -f1)
                local avg_duration=$(echo "$benchmark_details" | cut -d':' -f2)
                local preview=$(echo "$benchmark_details" | cut -d':' -f3-)
                echo "✅ Job completed (benchmark.raw.jsonl: ${row_count} rows, avg duration: ${avg_duration})"
                if [ -n "$preview" ] && [ "$preview" != "" ]; then
                    # Show preview on separate lines for better readability
                    echo "     📄 Preview:"
                    echo "$preview" | tr '|' '\n' | sed 's/^/     │ /'
                fi
            else
                echo "✅ Job completed (benchmark.raw.jsonl found)"
            fi
            ;;
        "COMPLETED")
            echo "✅ Job completed"
            ;;
        "STARTING")
            echo "🟡 Job starting (no logs directory yet)"
            ;;
        "STARTING:NO_LOGS")
            echo "🟡 Job starting (logs directory exists but no log files yet)"
            ;;
        "RUNNING:"*"min")
            local log_age="${status_info#RUNNING:}"
            echo "🟢 Job running (latest log updated ${log_age} ago)"
            ;;
        "RUNNING:NO_TIMESTAMP")
            echo "🟡 Job running (latest log has no timestamp)"
            ;;
        "STUCK:"*"min")
            local stuck_time="${status_info#STUCK:}"
            echo "🔴 Job STUCK (logs not updated for ${stuck_time})"
            ;;
        *)
            echo "❓ Unknown status: $status_info"
            ;;
    esac
}

# Main monitoring loop
main() {
    log "INFO" "Starting job stuck monitor"
    log "INFO" "Output directory: $OUTPUT_DIR_PREFIX"
    log "INFO" "Check interval: ${CHECK_INTERVAL_SECONDS}s"
    log "INFO" "Stuck threshold: ${STUCK_THRESHOLD_MINUTES}min"
    log "INFO" "Verbose mode: $([ $VERBOSE -eq 1 ] && echo "enabled" || echo "disabled")"
    echo ""
    
    local last_monitored_folder=""
    local last_status=""
    local stuck_warning_count=0
    
    while true; do
        # Find latest job folder
        local latest_folder=$(find_latest_job_folder "$OUTPUT_DIR_PREFIX")
        
        if [ $? -ne 0 ] || [ -z "$latest_folder" ]; then
            log "WARNING" "No job folder found, waiting..."
            sleep $CHECK_INTERVAL_SECONDS
            continue
        fi
        
        local folder_name=$(basename "$latest_folder")
        
        # Check if we're monitoring a new folder
        if [ "$latest_folder" != "$last_monitored_folder" ]; then
            log "INFO" "Now monitoring: $folder_name"
            last_monitored_folder="$latest_folder"
            stuck_warning_count=0
        fi
        
        # Analyze job status
        local current_status=$(analyze_job_for_stuck "$latest_folder")
        local status_display=$(get_status_display "$current_status")
        
        # Log status changes or periodic updates
        if [ "$current_status" != "$last_status" ]; then
            log "STATUS" "$folder_name - $status_display"
            last_status="$current_status"
            
            # Reset stuck warning count on status change
            if [[ "$current_status" != "STUCK:"* ]]; then
                stuck_warning_count=0
            fi
        elif [ $VERBOSE -eq 1 ]; then
            log "DEBUG" "$folder_name - $status_display"
        fi
        
        # Special handling for stuck jobs
        if [[ "$current_status" == "STUCK:"* ]]; then
            stuck_warning_count=$((stuck_warning_count + 1))
            
            # Alert every 5 checks (roughly 3-4 minutes) when stuck
            if [ $((stuck_warning_count % 5)) -eq 1 ]; then
                log "ALERT" "🚨 $folder_name - $status_display"
                echo "     Consider killing the job if it remains stuck"
            fi
        fi
        
        # Check if job completed
        if [[ "$current_status" == "COMPLETED" ]]; then
            log "INFO" "Job completed, continuing to monitor for new jobs..."
        fi
        
        sleep $CHECK_INTERVAL_SECONDS
    done
}

# Trap signals for clean exit
trap 'log "INFO" "Monitor stopped"; exit 0' SIGINT SIGTERM

# Start monitoring
main
