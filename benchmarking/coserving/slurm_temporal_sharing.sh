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
#SBATCH --job-name=temporal_sharing
#SBATCH --output=/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/slurm/%x_%A_%a.out
#SBATCH --error=/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/slurm/%x_%A_%a.err
#SBATCH --array=0-14   # 3 temporal × 3 models × 5 QPS = 45 jobs, max 10 running at once

set -xe

# Change directory to the script’s location relative to the build directory
CWD_="/global/homes/g/goliaro/flexllm"
cd "$CWD_/flexflow-serve/build"

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
TEMPORAL_SHARING_FREQUENCIES=(64)

# compute dimensions
model_count=${#MODEL_NAMES[@]}
qps_count=${#QPS_vals[@]}
temp_count=${#TEMPORAL_SHARING_FREQUENCIES[@]}
combos=$(( model_count * qps_count * temp_count ))

# map SLURM_ARRAY_TASK_ID → (t_idx, m_idx, q_idx)
idx=$SLURM_ARRAY_TASK_ID
t_idx=$(( idx / (model_count * qps_count) ))
rem=$(( idx % (model_count * qps_count) ))
m_idx=$(( rem / qps_count ))
q_idx=$(( rem % qps_count ))

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

echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] Running $MODEL_NAME on $trace with:"
echo "  TP=$NGPUS, TSF=$temporal_sharing_frequency, BZ=$BATCH_SIZE, TPB=$MAX_TOKENS_PER_BATCH, KV=$NUM_KV_CACHE_SLOTS, BWD=$NUM_BWD_LAYERS, QPS=$qps"

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
    2>&1 | tee "${OUTPUT_FOLDER}/logs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
