#!/bin/bash
set -e
set -x

cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

cd "${ROOT_DIR}/flexflow-serve/build"

TRACES_FOLDER="${ROOT_DIR}/traces"
kickstart_trace_file="${TRACES_FOLDER}/kickstart.json"
OUTPUT_FOLDER="${ROOT_DIR}/output/kickstart"
LOG_FILE="${OUTPUT_FOLDER}/logs/kickstart.log"

mkdir -p "$OUTPUT_FOLDER/output"
mkdir -p "$OUTPUT_FOLDER/logs"
mkdir -p "$OUTPUT_FOLDER/profiling"

export LEGION_BACKTRACE=1

TP_DEGREE=4

./inference/flexllm/peft_train \
    -ll:cpu 16 -ll:gpu $TP_DEGREE -ll:util 16 \
    -ll:fsize 37000 -ll:zsize 40000 -ll:csize 4096 \
    -llm-model meta-llama/Llama-3.1-8B-Instruct --fusion \
    -tensor-parallelism-degree $TP_DEGREE \
    -prompt "$kickstart_trace_file" \
    --peft-support-mode DISABLED \
    -output-file "${OUTPUT_FOLDER}/output/inference_eval.json" \
    -profiling-folder "$OUTPUT_FOLDER/profiling" \
    --max-requests-per-batch 256 \
    --max-tokens-per-batch 256 \
    --max-sequence-length 8192 \
    --num-kv-cache-slots 70000 \
    --ignore-eos 2>&1 > "$LOG_FILE"
