#!/bin/bash
#SBATCH --account=m4138
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --constraint gpu,ss11,a100,hbm80g
#SBATCH --time=02:00:00
#SBATCH --job-name=e2e_temporal_sharing
#SBATCH --job-name=flexflow_peft
#SBATCH --output=/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/slurm/%x_%A_%a.out
#SBATCH --error=/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/slurm/%x_%A_%a.err
#SBATCH --array=0-14

# Enable debugging and exit on error
set -x
set -e

# Change directory to the script’s location relative to the build directory
CWD_="/global/homes/g/goliaro/flexllm"
cd "$CWD_/flexflow-serve/build"

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
PEFT_SUPPORT_MODE="TEMPORAL_SHARING"

OUTPUT_FOLDER="../../benchmarking/output/e2e/temporal_sharing"
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

# --- Decode the SLURM array index ---
# Total iterations = number of models (3) x number of QPS values (5) = 15.
task_id=$SLURM_ARRAY_TASK_ID
model_index=$(( task_id / 5 ))
qps_index=$(( task_id % 5 ))

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

# Remove any previous output or log file; ignore errors if the files don’t exist
rm "$OUTPUT_FILE" "$LOG_FILE" 2>/dev/null || true

echo "Running $MODEL_NAME (tp=$NGPUS) on $trace with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS, NUM_BWD_LAYERS=$NUM_BWD_LAYERS, QPS=$qps, PEFT_SUPPORT_MODE=$PEFT_SUPPORT_MODE"

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