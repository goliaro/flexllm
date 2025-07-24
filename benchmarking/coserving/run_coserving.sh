#!/bin/bash

# Sequential version of the SLURM e2e_coserving script
# Runs all model/QPS combinations sequentially on a single node

# Enable debugging and exit on error
set -x
set -e

# Change directory to the script's location relative to the build directory
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"

# Set up the environment
source ./set_python_envs.sh

# Define the arrays from the original script
MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)
TP_DEGREES=(1 2 4)
ZSIZES=(40000 40000 70000)
NUM_BWD_LAYERS_vals=(2 1 1)
NUM_KV_CACHE_SLOTS_vals=(70000 70000 60000)
model_types=("llama" "qwen" "qwen")
QPS_vals=(5.0 4.0 3.0 2.0 1.0)

# Other parameters
NCPUS=16
FSIZE=77000
CSIZE=4096
MAX_SEQ_LEN=8192
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
MAX_TRAINING_EPOCHS=10000
GRADIENT_ACCUMULATION_STEPS=8
FT_LOGGING_STEPS=100
trace=sharegpt
PEFT_SUPPORT_MODE="COSERVING"

OUTPUT_FOLDER="../../benchmarking/output/e2e/coserving"
TRACES_FOLDER="../../benchmarking/traces/burstgpt"

# Create directories needed for outputs, logs, and profiling
mkdir -p "$OUTPUT_FOLDER/output"
mkdir -p "$OUTPUT_FOLDER/logs"
mkdir -p "$OUTPUT_FOLDER/profiling"

export LEGION_BACKTRACE=1
# Optionally, uncomment these for further debugging:
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_CPP_LOG_LEVEL=INFO
# export CUDA_LAUNCH_BLOCKING=1


# Function to count entries in the last JSON of a file
count_entries_in_last_json() {
    local filename="$1"
    
    # Check if file exists
    if [ ! -f "$filename" ]; then
        echo "Error: File '$filename' not found" >&2
        return 1
    fi
    
    # Extract all complete JSON objects from the file
    # This handles cases where JSONs might be on separate lines or concatenated
    local json_objects=$(cat "$filename" | jq -c '.')
    
    # Count the number of JSON objects
    local num_jsons=$(echo "$json_objects" | wc -l)
    
    if [ "$num_jsons" -eq 0 ]; then
        echo "Error: No valid JSON found in file" >&2
        return 1
    fi
    
    # Get the last JSON object
    local last_json=$(echo "$json_objects" | tail -n 1)
    
    # Count entries in the last JSON
    # Assumes the JSON is an array or has a top-level array field
    if echo "$last_json" | jq -e 'type == "array"' > /dev/null 2>&1; then
        # JSON is directly an array
        local count=$(echo "$last_json" | jq 'length')
    else
        # JSON is an object, look for array fields and use the first one found
        # If you know the specific field name, replace this with: jq '.your_field_name | length'
        local count=$(echo "$last_json" | jq '[.[] | select(type == "array")] | .[0] | length // 0')
    fi
    
    echo "$count"
    return 0
}

# Function to run a single experiment
run_experiment() {
    local model_index=$1
    local qps_index=$2
    
    MODEL_NAME=${MODEL_NAMES[$model_index]}
    PEFT_MODEL_NAME="${MODEL_NAME}-lora"
    NGPUS=${TP_DEGREES[$model_index]}
    ZSIZE=${ZSIZES[$model_index]}
    NUM_BWD_LAYERS=${NUM_BWD_LAYERS_vals[$model_index]}
    MODEL_TYPE=${model_types[$model_index]}
    NUM_KV_CACHE_SLOTS=${NUM_KV_CACHE_SLOTS_vals[$model_index]}
    TRACES_FOLDER_="../../benchmarking/traces/burstgpt/${MODEL_TYPE}"
    FINETUNING_DATASET="t1_${MODEL_TYPE}"
    FINETUNING_DATASET_FILE="${TRACES_FOLDER}/../${FINETUNING_DATASET}.json"

    qps=${QPS_vals[$qps_index]}
    TRACE_FILE="${TRACES_FOLDER_}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
    OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers_${qps}_qps_${PEFT_SUPPORT_MODE}.json"
    LOG_FILE="${OUTPUT_FOLDER}/logs/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers_${qps}_qps_${PEFT_SUPPORT_MODE}.log"

    # Check that the trace file and the finetuning dataset file exist
    if [[ ! -f "$TRACE_FILE" ]]; then
        echo "Trace file $TRACE_FILE does not exist. Skipping this experiment."
        exit 1
    fi
    if [[ ! -f "$FINETUNING_DATASET_FILE" ]]; then
        echo "Finetuning dataset file $FINETUNING_DATASET_FILE does not exist. Skipping this experiment."
        exit 1
    fi

    echo "========================================================================"
    echo "Running experiment $((model_index * 5 + qps_index + 1))/15"
    echo "Model: $MODEL_NAME (tp=$NGPUS)"
    echo "Trace: $trace"
    echo "Parameters: BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH"
    echo "KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS, NUM_BWD_LAYERS=$NUM_BWD_LAYERS"
    echo "QPS=$qps, PEFT_SUPPORT_MODE=$PEFT_SUPPORT_MODE"
    echo "========================================================================"

    # Check if the output file already exists and has enough entries, in which case we skip the experiment
    if [ -f "$OUTPUT_FILE" ]; then
        # Use a subshell to avoid exiting the script on errors from count_entries_in_last_json
        trace_entries=$(count_entries_in_last_json "$TRACE_FILE" 2>/dev/null)
        output_entries=$(count_entries_in_last_json "$OUTPUT_FILE" 2>/dev/null)
        
        required_entries=$(( trace_entries / 2 ))
        if [ "$output_entries" -ge "$required_entries" ]; then
            echo "Skipping experiment ${experiment_num}: Output file $OUTPUT_FILE exists with sufficient entries ($output_entries >= $required_entries)."
            echo ""
            return
        else
            echo "Rerunning experiment ${experiment_num}: Output file $OUTPUT_FILE exists but has insufficient entries ($output_entries < $required_entries)."
        fi
    fi

    # Execute the training/inference command
    ./inference/flexllm/peft_train \
        -ll:cpu "$NCPUS" -ll:gpu "$NGPUS" -ll:util "$NCPUS" \
        -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -ll:csize "$CSIZE" \
        -llm-model "$MODEL_NAME" --fusion \
        -tensor-parallelism-degree "$NGPUS" \
        -prompt "$TRACE_FILE" \
        -peft-model "$PEFT_MODEL_NAME" --peft-support-mode "$PEFT_SUPPORT_MODE" \
        -finetuning-dataset "$FINETUNING_DATASET_FILE" \
        --max-training-epochs "$MAX_TRAINING_EPOCHS" \
        --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
        --num-layers-per-finetuning-step "$NUM_BWD_LAYERS" \
        --num-logging-steps "$FT_LOGGING_STEPS" \
        -output-file "$OUTPUT_FILE" \
        -profiling-folder "${OUTPUT_FOLDER}/profiling" \
        --max-requests-per-batch "$BATCH_SIZE" \
        --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH" \
        --max-sequence-length "$MAX_SEQ_LEN" \
        --num-kv-cache-slots "$NUM_KV_CACHE_SLOTS" \
        --ignore-eos --warmup --log-instance-creation \
        2>&1 | tee "$LOG_FILE"
    
    echo "Completed experiment for $MODEL_NAME with QPS=$qps"
    echo ""
}

# Main execution: Loop through all combinations
echo "Starting sequential execution of all experiments..."
echo "Total experiments: $((${#MODEL_NAMES[@]} * ${#QPS_vals[@]}))"
echo ""

start_time=$(date +%s)

# Loop through all model indices
for model_index in $(seq 0 $((${#MODEL_NAMES[@]} - 1))); do
    # Loop through all QPS indices for each model
    for qps_index in $(seq 0 $((${#QPS_vals[@]} - 1))); do
        run_experiment $model_index $qps_index
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo "========================================================================"
echo "All experiments completed!"
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"
echo "Results can be found in: $OUTPUT_FOLDER"
echo "========================================================================"