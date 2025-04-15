set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"
source ./set_python_envs.sh

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
PEFT_MODEL_NAME="${MODEL_NAME}-lora"
TP_DEGREE=1
ZSIZE=40000
trace=sharegpt

QPS_vals=(
  # 5.0
  4.0
  3.0
  2.0
  1.0
)
NUM_BWD_LAYERS=2
NGPUS=1
NCPUS=16
FSIZE=77500
CSIZE=4096
MAX_SEQ_LEN=8192
NUM_KV_CACHE_SLOTS=90000
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256

MAX_TRAINING_EPOCHS=10000
GRADIENT_ACCUMULATION_STEPS=8
FT_LOGGING_STEPS=100

OUTPUT_FOLDER="../../benchmarking/output/coserving/flexllm/8B"
TRACES_FOLDER="../../benchmarking/traces/burstgpt/llama"
FINETUNING_DATASET="t1_llama"
FINETUNING_DATASET_FILE="${TRACES_FOLDER}/../../${FINETUNING_DATASET}.json"

mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling

export LEGION_BACKTRACE=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_CPP_LOG_LEVEL=INFO
# export CUDA_LAUNCH_BLOCKING=1

for qps in "${QPS_vals[@]}"; do
  TRACE_FILE="${TRACES_FOLDER}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
  OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}_${trace}_bz_${BATCH_SIZE}_tokens_per_batch_${MAX_TOKENS_PER_BATCH}_kv_cache_slots_${NUM_KV_CACHE_SLOTS}_${NUM_BWD_LAYERS}_bwd_layers.json"
  rm $OUTPUT_FILE || true
  
  echo "Running $MODEL_NAME (tp=$NGPUS) on $trace with BZ=$BATCH_SIZE, TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH, KV_CACHE_SLOTS=$NUM_KV_CACHE_SLOTS, NUM_BWD_LAYERS=$NUM_BWD_LAYERS"
  
  ./inference/flexllm/peft_train \
      -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
      -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
      -llm-model $MODEL_NAME --fusion \
      -tensor-parallelism-degree $NGPUS \
      -prompt $TRACE_FILE \
      -enable-peft -peft-model $PEFT_MODEL_NAME \
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
      --ignore-eos --warmup --log-instance-creation
done