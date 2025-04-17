#!/bin/bash
#SBATCH --account=m4138
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --constraint gpu,ss11,a100,hbm80g
#SBATCH --time=01:00:00
#SBATCH --job-name=vllm_bench
#SBATCH --output=/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/slurm/%x_%A_%a.out
#SBATCH --error=/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/slurm/%x_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --array=0-14

set -x
set -o pipefail

# Move to script dir and set python path
CWD_="/global/u2/g/goliaro/flexllm/benchmarking/vllm_online"
cd "$CWD_"
export PYTHONPATH="$(realpath $PWD/../../vllm)"


# Basic env
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_LOG_LEVEL="WARNING"

# Configuration (same as your original)
VLLM_V1=1
EAGER_MODE=true
declare -a MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)
declare -a TP_DEGREES=(1 2 4)
declare -a model_types=("llama" "qwen" "qwen")
declare -a QPS_vals=(20.0 16.0 12.0)
trace=sharegpt
BATCH_SIZE=256
MAX_TOKENS_PER_BATCH=256
MAX_NUM_REQUESTS=5000
MAX_SEQ_LEN=8192

# Helper functions (unchanged)
check_gpus() {
  gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}')
  echo "GPU type is $gpu_type"
}

wait_for_server() {
  timeout 1200 bash -c '
    until curl -X POST localhost:8000/v1/completions; do
      sleep 10
    done' && return 0 || return 1
}

kill_gpu_processes() {
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3   | xargs -r kill -9
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)" -ge 1000 ]; do
    sleep 1
  done
  rm -rf ~/.config/vllm
}

run_serving_tests() {
  local model_name=$1 tp_degree=$2 v1=$3 eager=$4 batch=$5 max_tok=$6 qps=$7 max_req=$8 trace=$9 trace_file=${10}
  if [ "$max_tok" -lt "$batch" ]; then
    echo "Skipping: max_tok < batch_size"
    return
  fi

  # launch server
  server_cmd="VLLM_USE_V1=${v1} python3 -m vllm.entrypoints.openai.api_server \
    --model ${model_name} \
    --tensor-parallel-size ${tp_degree} \
    --enable-chunked-prefill \
    --max-num-seqs ${batch} \
    --max-num-batched-tokens ${max_tok} \
    --disable-log-stats \
    --disable-log-requests \
    --swap-space 0"
  [ "$eager" = true ] && server_cmd+=" --enforce-eager"
  bash -c "${server_cmd}" & server_pid=$!

  if wait_for_server; then
    echo "Server is up."
  else
    echo "Server failed to start."
  fi

  mkdir -p ../output/vllm
  result_file=$(echo "results_${trace}_$( [ "$eager" = true ] && echo eager_ )$( [ "$v1" = 1 ] && echo v1_ )${model_name//\//_}_bz_${batch}_max_num_batched_tokens_${max_tok}_${qps}_qps.json" | tr '[:upper:]' '[:lower:]')

  client_cmd="VLLM_USE_V1=${v1} PYTHONPATH=${PYTHONPATH} python3 benchmark_vllm.py \
    --model ${model_name} \
    --backend vllm \
    --ignore-eos \
    --num-prompts ${max_req} \
    --dataset-path ${trace_file} \
    --save-result --save-detailed \
    --result-dir ../output/vllm \
    --result-filename ${result_file}"

  bash -c "${client_cmd}"

  kill -9 $server_pid
  kill_gpu_processes
}

# Derive which combo this task should run
N_QPS=${#QPS_vals[@]}
TASK_ID=$SLURM_ARRAY_TASK_ID
i=$(( TASK_ID / N_QPS ))
j=$(( TASK_ID % N_QPS ))

model_name=${MODEL_NAMES[$i]}
tp_degree=${TP_DEGREES[$i]}
qps=${QPS_vals[$j]}
trace_file="/global/homes/g/goliaro/flexllm/benchmarking/traces/burstgpt/${model_types[$i]}/${trace}_${MAX_SEQ_LEN}_${qps}_qps.json"

if [ ! -f "$trace_file" ]; then
  echo "Error: missing trace file: $trace_file"
  exit 1
fi

# Run exactly one test
check_gpus
run_serving_tests "$model_name" "$tp_degree" "$VLLM_V1" "$EAGER_MODE" "$BATCH_SIZE" "$MAX_TOKENS_PER_BATCH" "$qps" "$MAX_NUM_REQUESTS" "$trace" "$trace_file"