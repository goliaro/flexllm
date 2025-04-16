set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"
source ./set_python_envs.sh

MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)
TP_DEGREES=(
  1
  2
  4
)
ZSIZES=(
  40000
  40000
  70000
)
NUM_BWD_LAYERS_vals=(
  2
  1
  1
)
NUM_KV_CACHE_SLOTS_vals=(
  70000
  70000
  60000
)
model_types=(
  "llama"
  "qwen"
  "qwen"
)
QPS_vals=(
  5.0
  4.0
  3.0
  2.0
  1.0
)
NCPUS=16
FSIZE=77000
CSIZE=4096
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=60000
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
MAX_TRAINING_EPOCHS=10000
GRADIENT_ACCUMULATION_STEPS=8
FT_LOGGING_STEPS=10
trace=sharegpt

PEFT_SUPPORT_MODE="TEMPORAL_SHARING"

OUTPUT_FOLDER="../../benchmarking/output/e2e/temporal_sharing"
TRACES_FOLDER="../../benchmarking/traces/burstgpt"

mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling

export LEGION_BACKTRACE=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_CPP_LOG_LEVEL=INFO
# export CUDA_LAUNCH_BLOCKING=1

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    PEFT_MODEL_NAME="${MODEL_NAME}-lora"
    NGPUS=${TP_DEGREES[$i]}
    ZSIZE=${ZSIZES[$i]}
    NUM_BWD_LAYERS=${NUM_BWD_LAYERS_vals[$i]}
    MODEL_TYPE=${model_types[$i]}
    NUM_KV_CACHE_SLOTS=${NUM_KV_CACHE_SLOTS_vals[$i]}
    TRACES_FOLDER_="../../benchmarking/traces/burstgpt/${MODEL_TYPE}"
    FINETUNING_DATASET="t1_${MODEL_TYPE}"
    FINETUNING_DATASET_FILE="${TRACES_FOLDER}/../${FINETUNING_DATASET}.json"

    for qps in "${QPS_vals[@]}"; do
        TRACE_FILE="${TRACES_FOLDER_}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
        OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers_${qps}_qps_${PEFT_SUPPORT_MODE}.json"
        LOG_FILE="${OUTPUT_FOLDER}/logs/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers_${qps}_qps_${PEFT_SUPPORT_MODE}.log"
        rm $OUTPUT_FILE $LOG_FILE || true
        
        echo "Running $MODEL_NAME (tp=$NGPUS) on $trace with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS, NUM_BWD_LAYERS=$NUM_BWD_LAYERS, QPS=$qps, PEFT_SUPPORT_MODE=$PEFT_SUPPORT_MODE"
        
        ./inference/flexllm/peft_train \
            -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
            -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
            -llm-model $MODEL_NAME --fusion \
            -tensor-parallelism-degree $NGPUS \
            -prompt $TRACE_FILE \
            -peft-model $PEFT_MODEL_NAME --peft-support-mode $PEFT_SUPPORT_MODE \
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
            2>&1 | tee $LOG_FILE
    done
done