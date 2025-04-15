#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../../flexflow-serve/build"

# reset
# ../config/config.linux
# make -j 
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

NCPUS=16
FSIZE=77000
ZSIZE=80000
CSIZE=4096
MAX_SEQ_LEN=8192
MAX_REQUESTS_PER_BATCH=256
MAX_TOKENS_PER_BATCH=1024
NUM_KV_CACHE_SLOTS=60000

MAX_FINETUNING_FWD_TOKENS=(
    "0,128,256,384,512"
)
MAX_FINETUNING_BWD_LAYERS="0,8,16,24,32"

OUTPUT_FOLDER="../../benchmarking/output/overhead_test"
mkdir -p $OUTPUT_FOLDER/output
mkdir -p $OUTPUT_FOLDER/logs
mkdir -p $OUTPUT_FOLDER/profiling 


# python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download(repo_id=\"${MODEL_NAME}\", allow_patterns=\"*.safetensors\", max_workers=30)"
# python ../inference/utils/download_hf_model.py $MODEL_NAME --half-precision-only
# python ../inference/utils/download_peft_model.py $PEFT_MODEL_NAME --half-precision-only
 
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_FILE=/usr/FlexFlow/inference/output/nccl2.log
export LEGION_BACKTRACE=1
# export CUDA_VISIBLE_DEVICES=1,2,3,4

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    NGPUS=${TP_DEGREES[$i]}
    OUTPUT_FILE="${OUTPUT_FOLDER}/output/${MODEL_NAME//\//_}.json"
    LOG_FILE="${OUTPUT_FOLDER}/logs/${MODEL_NAME//\//_}.log"
    rm $LOG_FILE $OUTPUT_FILE || true

    ./inference/flexllm/overhead_test \
        -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE \
        -llm-model $MODEL_NAME --fusion  \
        -peft-model $PEFT_MODEL_NAME \
        -tensor-parallelism-degree $NGPUS \
        -output-file $OUTPUT_FILE \
        -profiling-folder "${OUTPUT_FOLDER}/profiling" \
        --max-requests-per-batch $MAX_REQUESTS_PER_BATCH \
        --max-tokens-per-batch $MAX_TOKENS_PER_BATCH \
        --max-sequence-length $MAX_SEQ_LEN \
        --max-fwd-finetuning-tokens $MAX_FINETUNING_FWD_TOKENS \
        --num-layers-per-finetuning-step $MAX_FINETUNING_BWD_LAYERS \
        --num-kv-cache-slots $NUM_KV_CACHE_SLOTS \
        2>&1 | tee $LOG_FILE
done
