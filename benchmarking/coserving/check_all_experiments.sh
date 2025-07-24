#!/bin/bash

# Script to check completion status of all experiments across co-serving, spatial sharing, and temporal sharing
# Reports completed experiments and lists those that are missing or incomplete

set -e

# Change to the build directory (same as the other scripts)
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"

# Common parameters (from the original scripts)
MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)
TP_DEGREES=(1 2 4)
model_types=("llama" "qwen" "qwen")
QPS_vals=(5.0 4.0 3.0 2.0 1.0)
TEMPORAL_SHARING_FREQUENCIES=(64 128 512)

# Parameters specific to each mode
COSERVING_NUM_BWD_LAYERS_vals=(2 1 1)
SPATIAL_NUM_BWD_LAYERS_vals=(1 1 1)
TEMPORAL_NUM_BWD_LAYERS_vals=(-1 -1 -1)

ZSIZES=(40000 40000 70000)
NUM_KV_CACHE_SLOTS_vals=(70000 70000 60000)

# Fixed parameters
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
MAX_SEQ_LEN=8192
trace="sharegpt"

# Function to count entries in the last JSON of a file (copied from original scripts)
count_entries_in_last_json() {
    local filename="$1"
    
    # Check if file exists
    if [ ! -f "$filename" ]; then
        echo "0"
        return 0
    fi
    
    # Extract all complete JSON objects from the file
    local json_objects=$(cat "$filename" | jq -c '.' 2>/dev/null)
    
    # Count the number of JSON objects
    local num_jsons=$(echo "$json_objects" | wc -l 2>/dev/null)
    
    if [ "$num_jsons" -eq 0 ]; then
        echo "0"
        return 0
    fi
    
    # Get the last JSON object
    local last_json=$(echo "$json_objects" | tail -n 1 2>/dev/null)
    
    # Count entries in the last JSON
    if echo "$last_json" | jq -e 'type == "array"' > /dev/null 2>&1; then
        # JSON is directly an array
        local count=$(echo "$last_json" | jq 'length' 2>/dev/null || echo "0")
    else
        # JSON is an object, look for array fields and use the first one found
        local count=$(echo "$last_json" | jq '[.[] | select(type == "array")] | .[0] | length // 0' 2>/dev/null || echo "0")
    fi
    
    echo "$count"
    return 0
}

# Function to check if an experiment is complete
check_experiment_complete() {
    local trace_file="$1"
    local output_file="$2"
    
    if [ ! -f "$trace_file" ]; then
        echo "TRACE_MISSING"
        return 0
    fi
    
    if [ ! -f "$output_file" ]; then
        echo "OUTPUT_MISSING"
        return 0
    fi
    
    local trace_entries=$(count_entries_in_last_json "$trace_file")
    local output_entries=$(count_entries_in_last_json "$output_file")
    local required_entries=$(( trace_entries / 2 ))
    
    if [ "$output_entries" -ge "$required_entries" ]; then
        echo "COMPLETE"
    else
        echo "INCOMPLETE:$output_entries/$required_entries"
    fi
}

# Initialize counters
total_experiments=0
completed_experiments=0
missing_output=0
incomplete_experiments=0
missing_trace=0

declare -a incomplete_list
declare -a missing_output_list
declare -a missing_trace_list

echo "========================================================================"
echo "Checking experiment completion status..."
echo "========================================================================"

# Check Co-serving experiments
echo "Checking Co-serving experiments..."
OUTPUT_FOLDER="../../benchmarking/output/e2e/coserving"
PEFT_SUPPORT_MODE="COSERVING"

for model_index in $(seq 0 $((${#MODEL_NAMES[@]} - 1))); do
    MODEL_NAME=${MODEL_NAMES[$model_index]}
    MODEL_TYPE=${model_types[$model_index]}
    NUM_BWD_LAYERS=${COSERVING_NUM_BWD_LAYERS_vals[$model_index]}
    NUM_KV_CACHE_SLOTS=${NUM_KV_CACHE_SLOTS_vals[$model_index]}
    TRACES_FOLDER_="../../benchmarking/traces/burstgpt/${MODEL_TYPE}"
    
    for qps_index in $(seq 0 $((${#QPS_vals[@]} - 1))); do
        qps=${QPS_vals[$qps_index]}
        TRACE_FILE="${TRACES_FOLDER_}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
        OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers_${qps}_qps_${PEFT_SUPPORT_MODE}.json"
        
        status=$(check_experiment_complete "$TRACE_FILE" "$OUTPUT_FILE")
        total_experiments=$((total_experiments + 1))
        
        case "$status" in
            "COMPLETE")
                completed_experiments=$((completed_experiments + 1))
                ;;
            "OUTPUT_MISSING")
                missing_output=$((missing_output + 1))
                missing_output_list+=("COSERVING: ${MODEL_NAME} QPS=${qps}")
                ;;
            "TRACE_MISSING")
                missing_trace=$((missing_trace + 1))
                missing_trace_list+=("COSERVING: ${MODEL_NAME} QPS=${qps} (trace: $TRACE_FILE)")
                ;;
            INCOMPLETE:*)
                incomplete_experiments=$((incomplete_experiments + 1))
                entries_info=${status#INCOMPLETE:}
                incomplete_list+=("COSERVING: ${MODEL_NAME} QPS=${qps} ($entries_info entries)")
                ;;
        esac
    done
done

# Check Spatial Sharing experiments
echo "Checking Spatial Sharing experiments..."
OUTPUT_FOLDER="../../benchmarking/output/e2e/spatial_sharing"
PEFT_SUPPORT_MODE="SPATIAL_SHARING_LIMITED"

for model_index in $(seq 0 $((${#MODEL_NAMES[@]} - 1))); do
    MODEL_NAME=${MODEL_NAMES[$model_index]}
    MODEL_TYPE=${model_types[$model_index]}
    NUM_BWD_LAYERS=${SPATIAL_NUM_BWD_LAYERS_vals[$model_index]}
    NUM_KV_CACHE_SLOTS=${NUM_KV_CACHE_SLOTS_vals[$model_index]}
    TRACES_FOLDER_="../../benchmarking/traces/burstgpt/${MODEL_TYPE}"
    
    for qps_index in $(seq 0 $((${#QPS_vals[@]} - 1))); do
        qps=${QPS_vals[$qps_index]}
        TRACE_FILE="${TRACES_FOLDER_}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
        OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers_${qps}_qps_${PEFT_SUPPORT_MODE}.json"
        
        status=$(check_experiment_complete "$TRACE_FILE" "$OUTPUT_FILE")
        total_experiments=$((total_experiments + 1))
        
        case "$status" in
            "COMPLETE")
                completed_experiments=$((completed_experiments + 1))
                ;;
            "OUTPUT_MISSING")
                missing_output=$((missing_output + 1))
                missing_output_list+=("SPATIAL: ${MODEL_NAME} QPS=${qps}")
                ;;
            "TRACE_MISSING")
                missing_trace=$((missing_trace + 1))
                missing_trace_list+=("SPATIAL: ${MODEL_NAME} QPS=${qps} (trace: $TRACE_FILE)")
                ;;
            INCOMPLETE:*)
                incomplete_experiments=$((incomplete_experiments + 1))
                entries_info=${status#INCOMPLETE:}
                incomplete_list+=("SPATIAL: ${MODEL_NAME} QPS=${qps} ($entries_info entries)")
                ;;
        esac
    done
done

# Check Temporal Sharing experiments
echo "Checking Temporal Sharing experiments..."
PEFT_SUPPORT_MODE="TEMPORAL_SHARING"

for t_idx in $(seq 0 $((${#TEMPORAL_SHARING_FREQUENCIES[@]} - 1))); do
    temporal_sharing_frequency=${TEMPORAL_SHARING_FREQUENCIES[$t_idx]}
    OUTPUT_FOLDER="../../benchmarking/output/e2e/temporal_sharing/${temporal_sharing_frequency}"
    
    for model_index in $(seq 0 $((${#MODEL_NAMES[@]} - 1))); do
        MODEL_NAME=${MODEL_NAMES[$model_index]}
        MODEL_TYPE=${model_types[$model_index]}
        NUM_BWD_LAYERS=${TEMPORAL_NUM_BWD_LAYERS_vals[$model_index]}
        NUM_KV_CACHE_SLOTS=${NUM_KV_CACHE_SLOTS_vals[$model_index]}
        TRACES_FOLDER_="../../benchmarking/traces/burstgpt/${MODEL_TYPE}"
        
        for qps_index in $(seq 0 $((${#QPS_vals[@]} - 1))); do
            qps=${QPS_vals[$qps_index]}
            TRACE_FILE="${TRACES_FOLDER_}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
            OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${qps}_qps_${PEFT_SUPPORT_MODE}.json"
            
            status=$(check_experiment_complete "$TRACE_FILE" "$OUTPUT_FILE")
            total_experiments=$((total_experiments + 1))
            
            case "$status" in
                "COMPLETE")
                    completed_experiments=$((completed_experiments + 1))
                    ;;
                "OUTPUT_MISSING")
                    missing_output=$((missing_output + 1))
                    missing_output_list+=("TEMPORAL(TSF=${temporal_sharing_frequency}): ${MODEL_NAME} QPS=${qps}")
                    ;;
                "TRACE_MISSING")
                    missing_trace=$((missing_trace + 1))
                    missing_trace_list+=("TEMPORAL(TSF=${temporal_sharing_frequency}): ${MODEL_NAME} QPS=${qps} (trace: $TRACE_FILE)")
                    ;;
                INCOMPLETE:*)
                    incomplete_experiments=$((incomplete_experiments + 1))
                    entries_info=${status#INCOMPLETE:}
                    incomplete_list+=("TEMPORAL(TSF=${temporal_sharing_frequency}): ${MODEL_NAME} QPS=${qps} ($entries_info entries)")
                    ;;
            esac
        done
    done
done

echo "========================================================================"
echo "EXPERIMENT STATUS SUMMARY"
echo "========================================================================"
echo "Total experiments expected: $total_experiments"
echo "Completed experiments: $completed_experiments"
echo "Missing output files: $missing_output"
echo "Incomplete experiments: $incomplete_experiments"
echo "Missing trace files: $missing_trace"
echo ""

completion_rate=$(( (completed_experiments * 100) / total_experiments ))
echo "Completion rate: ${completion_rate}%"
echo ""

if [ $missing_trace -gt 0 ]; then
    echo "========================================================================"
    echo "EXPERIMENTS WITH MISSING TRACE FILES ($missing_trace):"
    echo "========================================================================"
    for item in "${missing_trace_list[@]}"; do
        echo "  $item"
    done
    echo ""
fi

if [ $missing_output -gt 0 ]; then
    echo "========================================================================"
    echo "EXPERIMENTS WITH MISSING OUTPUT FILES ($missing_output):"
    echo "========================================================================"
    for item in "${missing_output_list[@]}"; do
        echo "  $item"
    done
    echo ""
fi

if [ $incomplete_experiments -gt 0 ]; then
    echo "========================================================================"
    echo "INCOMPLETE EXPERIMENTS ($incomplete_experiments):"
    echo "========================================================================"
    for item in "${incomplete_list[@]}"; do
        echo "  $item"
    done
    echo ""
fi

if [ $completed_experiments -eq $total_experiments ]; then
    echo "✅ All experiments completed successfully!"
else
    incomplete_total=$((missing_output + incomplete_experiments + missing_trace))
    echo "❌ $incomplete_total experiments need attention"
fi

echo "========================================================================"