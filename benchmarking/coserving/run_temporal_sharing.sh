#!/bin/bash

# Sequential version of the SLURM temporal_sharing script
# Runs all temporal frequency/model/QPS combinations sequentially on a single node

set -xe

# Change directory to the script's location relative to the build directory
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"

source ./set_python_envs.sh

# --- static parameters ---
NCPUS=16
FSIZE=77000
CSIZE=4096
MAX_SEQ_LEN=8192
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
MAX_TRAINING_EPOCHS=10000
GRADIENT_ACCUMULATION_STEPS=8
FT_LOGGING_STEPS=1
PEFT_SUPPORT_MODE="TEMPORAL_SHARING"
trace=sharegpt

# --- arrays to sweep ---
MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)
TP_DEGREES=(1 2 4)
ZSIZES=(40000 40000 70000)
NUM_BWD_LAYERS_vals=(-1 -1 -1)
NUM_KV_CACHE_SLOTS_vals=(70000 70000 60000)
model_types=(llama qwen qwen)
QPS_vals=(5.0 4.0 3.0 2.0 1.0)
TEMPORAL_SHARING_FREQUENCIES=(64 128 512)

export LEGION_BACKTRACE=1

# compute dimensions
model_count=${#MODEL_NAMES[@]}
qps_count=${#QPS_vals[@]}
temp_count=${#TEMPORAL_SHARING_FREQUENCIES[@]}
combos=$(( model_count * qps_count * temp_count ))

echo "Starting sequential execution of all temporal sharing experiments..."
echo "Total experiments: $combos (${temp_count} temporal frequencies × ${model_count} models × ${qps_count} QPS values)"
echo ""

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
    local t_idx=$1
    local m_idx=$2
    local q_idx=$3
    local experiment_num=$4
    
    # pick parameters for this task
    temporal_sharing_frequency=${TEMPORAL_SHARING_FREQUENCIES[$t_idx]}
    MODEL_NAME=${MODEL_NAMES[$m_idx]}
    NGPUS=${TP_DEGREES[$m_idx]}
    ZSIZE=${ZSIZES[$m_idx]}
    NUM_BWD_LAYERS=${NUM_BWD_LAYERS_vals[$m_idx]}
    NUM_KV_CACHE_SLOTS=${NUM_KV_CACHE_SLOTS_vals[$m_idx]}
    MODEL_TYPE=${model_types[$m_idx]}
    qps=${QPS_vals[$q_idx]}

    # set up folders
    OUTPUT_FOLDER="../../benchmarking/output/e2e/temporal_sharing/${temporal_sharing_frequency}"
    TRACES_FOLDER="../../benchmarking/traces/burstgpt/${MODEL_TYPE}"
    FINETUNING_DATASET="t1_${MODEL_TYPE}"
    FINETUNING_DATASET_FILE="${TRACES_FOLDER}/../../${FINETUNING_DATASET}.json"
    TRACE_FILE="${TRACES_FOLDER}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"

    mkdir -p "${OUTPUT_FOLDER}/output" "${OUTPUT_FOLDER}/profiling" "${OUTPUT_FOLDER}/logs"

    OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${qps}_qps_${PEFT_SUPPORT_MODE}.json"
    LOG_FILE="${OUTPUT_FOLDER}/logs/experiment_${experiment_num}_t${t_idx}_m${m_idx}_q${q_idx}.log"

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
    echo "Running experiment ${experiment_num}/${combos}"
    echo "Temporal Sharing Frequency: $temporal_sharing_frequency"
    echo "Model: $MODEL_NAME (tp=$NGPUS)"
    echo "Trace: $trace"
    echo "Parameters: TP=$NGPUS, TSF=$temporal_sharing_frequency, BZ=$BATCH_SIZE"
    echo "TPB=$MAX_TOKENS_PER_BATCH, KV=$NUM_KV_CACHE_SLOTS, BWD=$NUM_BWD_LAYERS, QPS=$qps"
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
    
    ./inference/flexllm/peft_train \
        -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
        -llm-model $MODEL_NAME --fusion \
        -tensor-parallelism-degree $NGPUS \
        -prompt $TRACE_FILE \
        -peft-model "${MODEL_NAME}-lora" \
        --peft-support-mode $PEFT_SUPPORT_MODE \
        --temporal-sharing-frequency $temporal_sharing_frequency \
        -finetuning-dataset $FINETUNING_DATASET_FILE \
        --max-training-epochs $MAX_TRAINING_EPOCHS \
        --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
        --num-layers-per-finetuning-step $NUM_BWD_LAYERS \
        --num-logging-steps $FT_LOGGING_STEPS \
        -output-file $OUTPUT_FILE \
        -profiling-folder "${OUTPUT_FOLDER}/profiling" \
        --max-requests-per-batch $BATCH_SIZE \
        --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
        --max-sequence-length $MAX_SEQ_LEN \
        --num-kv-cache-slots $NUM_KV_CACHE_SLOTS \
        --ignore-eos --log-instance-creation \
        2>&1 | tee "$LOG_FILE"
    
    echo "Completed experiment ${experiment_num}: $MODEL_NAME with TSF=$temporal_sharing_frequency, QPS=$qps"
    echo ""
}

# Main execution: Loop through all combinations
start_time=$(date +%s)
experiment_num=1

# Loop through temporal sharing frequencies
for t_idx in $(seq 0 $((temp_count - 1))); do
    echo "Processing temporal sharing frequency: ${TEMPORAL_SHARING_FREQUENCIES[$t_idx]}"
    
    # Loop through all models
    for m_idx in $(seq 0 $((model_count - 1))); do
        echo "  Processing model: ${MODEL_NAMES[$m_idx]} (index $m_idx)"
        
        # Loop through all QPS values for each model
        for q_idx in $(seq 0 $((qps_count - 1))); do
            run_experiment $t_idx $m_idx $q_idx $experiment_num
            experiment_num=$((experiment_num + 1))
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo "========================================================================"
echo "All temporal sharing experiments completed!"
echo "Total experiments run: $((experiment_num - 1))"
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"
echo "Results can be found in: ../../benchmarking/output/e2e/temporal_sharing/"
echo "========================================================================"