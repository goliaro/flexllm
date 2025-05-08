#!/bin/bash

set -x
set -o pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"
export PYTHONPATH="$(realpath $PWD/../../vllm)"

VLLM_V1=1
EAGER_MODE=true
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
model_types=(
  "llama"
  "qwen"
  "qwen"
)
QPS_vals=(
  6.7 # 20/3
  5.3 # 16/3
  2.7 # 8/3
  1.3 # 4/3
  20.0
  16.0
  12.0
  10.0
  8.0
  6.0
  4.0
  2.0
)
trace=sharegpt
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
MAX_NUM_REQUESTS=5000
MAX_SEQ_LEN=8192

check_gpus() {
  # check the number of GPUs and GPU type.
  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
  echo "GPU type is $gpu_type"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  timeout 1200 bash -c '
    until curl -X POST localhost:8000/v1/completions; do
      sleep 10
    done' && return 0 || return 1
}

kill_gpu_processes() {
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9


  # wait until GPU memory usage smaller than 1GB
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
    sleep 1
  done

  # remove vllm config file
  rm -rf ~/.config/vllm

}

run_serving_tests() {
  local model_name=${1}
  local tp_degree=${2}
  local vllm_use_v1=${3}
  local eager_mode=${4}
  local batch_size=${5}
  local max_num_batched_tokens=${6}
  local qps=${7}
  local max_num_requests=${8}
  local trace=${9}
  local trace_file=${10}

  # if the max_num_batched_tokens is less than the batch_size, return
  if [ "$max_num_batched_tokens" -lt "$batch_size" ]; then
    echo "max_num_batched_tokens is less than batch_size, skipping this test."
    return
  fi

  server_command="VLLM_USE_V1=${vllm_use_v1} python3 \
      -m vllm.entrypoints.openai.api_server \
      --model ${model_name} \
      --tensor-parallel-size ${tp_degree} \
      --enable-chunked-prefill \
      --max-num-seqs ${batch_size} \
      --max-num-batched-tokens ${max_num_batched_tokens} \
      --disable-log-stats \
      --disable-log-requests
      --swap-space 0"
  
  if [ "$eager_mode" = true ]; then
    server_command+=" --enforce-eager"
  fi
  echo "Starting VLLM server"
  echo "Server command: $server_command"
  bash -c "$server_command" &
  server_pid=$!

  # wait until the server is alive
  if wait_for_server; then
    echo ""
    echo "vllm server is up and running."
  else
    echo ""
    echo "vllm failed to start within the timeout period."
  fi

  mkdir -p ../output/vllm

  # Construct the result filename and convert it to lowercase.
  result_filename=$(echo "results_${trace}_$( [ "$eager_mode" = true ] && echo "eager_" )$( [ "$vllm_use_v1" = 1 ] && echo "v1_" )${model_name//\//_}_bz_${batch_size}_max_num_batched_tokens_${max_num_batched_tokens}_${qps}_qps_.json" | tr '[:upper:]' '[:lower:]')

  # Build the client command with the result_filename variable.
  client_command="VLLM_USE_V1=${vllm_use_v1} PYTHONPATH=${PYTHONPATH} python3 benchmark_vllm.py \
        --model ${model_name} \
        --backend vllm \
        --ignore-eos \
        --num-prompts ${max_num_requests} \
        --dataset-path ${trace_file} \
        --save-result --save-detailed \
        --result-dir ../output/vllm \
        --result-filename ${result_filename}"

  echo "Client command: $client_command"
  bash -c "$client_command"

  # clean up
  kill -9 $server_pid
  kill_gpu_processes
}

main() {
    check_gpus
    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    (which lsof) || (apt-get update && apt-get install -y lsof)

    # get the current IP address, required by benchmark_serving.py
    export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
    # turn of the reporting of the status of each request, to clean up the terminal output
    export VLLM_LOG_LEVEL="WARNING"

    for i in "${!MODEL_NAMES[@]}"; do
        for qps in "${QPS_vals[@]}"; do
            model_name="${MODEL_NAMES[$i]}"
            tp_degree="${TP_DEGREES[$i]}"
            MODEL_TYPE=${model_types[$i]}
            trace_file="/global/homes/g/goliaro/flexllm/benchmarking/traces/burstgpt/${MODEL_TYPE}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"
            # Check if the trace file exists
            if [ ! -f "$trace_file" ]; then
              echo "Error: Trace file $trace_file does not exist!"
              exit 1
            fi
            run_serving_tests "$model_name" "$tp_degree" "$VLLM_V1" "$EAGER_MODE" "$BATCH_SIZE" "$MAX_TOKENS_PER_BATCH" $qps $MAX_NUM_REQUESTS "$trace" $trace_file
        done
    done

}

main
