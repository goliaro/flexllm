#!/bin/bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

MODEL_NAMES=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
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

for i in "${!MODEL_NAMES[@]}"; do
  for qps in "${QPS_vals[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    echo "Running trace generation for model: $model_name at QPS: $qps"
    python get_burstgpt_trace.py --model_name "$model_name" --qps "$qps" & 
  done
done

wait
