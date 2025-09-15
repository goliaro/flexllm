#!/bin/bash
set -e
# set -x

cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)
TRACES_FOLDER="${ROOT_DIR}/traces"
OUTPUT_FOLDER="${ROOT_DIR}/output/kickstart"
LLAMA_FACTORY_FOLDER="${ROOT_DIR}/LLaMA-Factory"
rm -rf "$OUTPUT_FOLDER"
mkdir -p "$OUTPUT_FOLDER"
#################### Preliminary checks ####################

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to get user confirmation
get_confirmation() {
  local prompt="$1"
  local response
  
  while true; do
    echo -e "${YELLOW}$prompt${NC}"
    read -p "(y/N): " response
    
    case "$response" in
      y|yes|Y|YES)
        return 0
        ;;
      *)
        echo -e "${RED}Setup verification failed. Please complete the required steps before proceeding.${NC}"
        exit 1
        ;;
    esac
  done
}

echo ""
echo -e "${GREEN}=== FlexLLM Setup Verification ===${NC}"
echo "This script will verify that you have completed all the preparation steps."
echo "Please answer 'y', 'yes', 'Y', or 'YES' to confirm each step."
echo ""

# Step 1: Build container
get_confirmation "Have you run './docker/build_container.sh' to build the Docker container?"

# Step 2: Start container
get_confirmation "Have you run './docker/start_container.sh' to start the container (it should be running in the background)?"

get_confirmation "Have you run './docker/setup_flexllm.sh' and provided your Huggingface token when prompted?"

# Step 4: Attach to container
get_confirmation "Have you run './docker/attach_to_container.sh' to open a terminal connected to the container?"

echo ""
echo -e "${GREEN}✓ Checklist complete!${NC}"
# echo -e "${GREEN}You should now be ready to run FlexLLM experiments.${NC}"
echo ""

echo -e "${YELLOW}Running sanity checks...${NC}"

# # Check that the script is run from within docker
# if [ ! -f /.dockerenv ]; then
#     echo -e "${RED}Error: Please run this script from within the Docker container.${NC}"
#     exit 1
# fi

# Check that the gpus are available
gpu_count=$(nvidia-smi -L | wc -l)
if [ "$gpu_count" -lt 4 ]; then
  echo -e "${RED}Error: At least 4 NVIDIA GPUs are required. Found $gpu_count.${NC}"
  exit 1
fi

# Verify that all GPUs are A100s to ensure consistency with the paper
while IFS= read -r gpu; do
  if [[ "$gpu" != *"A100"* ]]; then
    echo -e "${RED}Error: GPU '$gpu' is not an A100. Please ensure all GPUs are NVIDIA A100 models.${NC}"
    exit 1
  fi
done < <(nvidia-smi --query-gpu=name --format=csv,noheader)

# Extra sanity check: Verify that each GPU has at least 80GB (81920 MB) of memory
for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id")
  if [ "$total_mem" -lt 81920 ]; then
    echo -e "${RED}Error: GPU $gpu_id has only ${total_mem} MB of memory. At least 80GB (81920 MB) is required.${NC}"
    exit 1
  fi
done



# Check that the traces are available
if [ ! -d $TRACES_FOLDER ]; then
  echo -e "${RED}Error: Traces directory not found. Please run get_traces.sh first.${NC}"
  exit 1
fi

# check that traces actually exist
check_trace_file() {
  local filepath="$1"
  if [ ! -f "$filepath" ]; then
    echo -e "${RED}Error: Trace file $filepath not found. Please run get_traces.sh first.${NC}"
    exit 1
  fi
}

QPS_vals=(
  # VLLM/FlexLLM QPS values
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
  # FlexLLM QPS values
  1.0
  3.0
  5.0
)
for qps in "${QPS_vals[@]}"; do
  file1="${TRACES_FOLDER}/burstgpt/llama/sharegpt_8192_${qps}_qps.json"
  file2="${TRACES_FOLDER}/burstgpt/qwen/sharegpt_8192_${qps}_qps.json"
  
  check_trace_file "$file1"
  check_trace_file "$file2"
done
# Finetuning traces
t1_llama_file="${TRACES_FOLDER}/t1_llama.json"
t1_qwen_file="${TRACES_FOLDER}/t1_qwen.json"
t1_llama_factory_file="${LLAMA_FACTORY_FOLDER}/data/t1_flexllm_llama.json"
t1_qwen_factory_file="${LLAMA_FACTORY_FOLDER}/data/t1_flexllm_qwen.json"
check_trace_file "$t1_llama_file"
check_trace_file "$t1_qwen_file"
check_trace_file "$t1_llama_factory_file"
check_trace_file "$t1_qwen_factory_file"
# Kickstart trace
kickstart_trace_file="${TRACES_FOLDER}/kickstart.json"
check_trace_file "$kickstart_trace_file"

echo -e "${GREEN}✓ All sanity checks passed!${NC}"
echo ""

check_output_file() {
  local filepath="$1"
  if [ ! -f "$filepath" ]; then
    echo -e "${RED}Error: Expected output file $filepath not found. Check the terminal and the logs (${OUTPUT_FOLDER}/logs) for any errors.${NC}"
    exit 1
  fi
}

#################### Test co-serving ####################
LOG_FILE="${OUTPUT_FOLDER}/logs/kickstart.log"
echo -e "${YELLOW}Testing co-serving...${NC}"
echo "Output redirected to $LOG_FILE"
echo ""

cd "${ROOT_DIR}/flexflow-serve/build"
source ./set_python_envs.sh

mkdir -p "$OUTPUT_FOLDER/output"
mkdir -p "$OUTPUT_FOLDER/logs"
mkdir -p "$OUTPUT_FOLDER/profiling"

export LEGION_BACKTRACE=1
./inference/flexllm/peft_train \
    -ll:cpu 16 -ll:gpu 1 -ll:util 16 \
    -ll:fsize 77000 -ll:zsize 40000 -ll:csize 4096 \
    -llm-model meta-llama/Llama-3.1-8B-Instruct --fusion \
    -tensor-parallelism-degree 1 \
    -prompt "$kickstart_trace_file" \
    -peft-model kickstart-lora --peft-support-mode COSERVING \
    -finetuning-dataset "${TRACES_FOLDER}/t1_llama.json" \
    --max-training-epochs 10000 \
    --gradient-accumulation-steps 8 \
    --num-layers-per-finetuning-step 2 \
    --num-logging-steps 100 \
    -output-file "${OUTPUT_FOLDER}/output/coserving.json" \
    -profiling-folder "$OUTPUT_FOLDER/profiling" \
    --max-requests-per-batch 256 \
    --max-tokens-per-batch 256 \
    --max-sequence-length 8192 \
    --num-kv-cache-slots 70000 \
    --ignore-eos --warmup --log-instance-creation \
    2>&1 > "$LOG_FILE"

check_output_file "${OUTPUT_FOLDER}/output/coserving.json"
echo -e "${GREEN}✓ Co-serving test passed!${NC}"
echo ""

##################################################

#################### Test vLLM ####################
echo -e "${YELLOW}Testing vLLM...${NC}"
cd "${ROOT_DIR}/benchmarking/vllm_online"

wait_for_server() {
  local max_attempts=120  # 120 * 10 seconds = 1200 seconds
  local attempt=0
  
  while [ $attempt -lt $max_attempts ]; do
    if curl -s -X POST localhost:8000/v1/completions >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
    ((attempt++))
  done
  return 1
}

kill_gpu_processes() {
  lsof -t -i:8000 | xargs -r kill -9 2>/dev/null || true
  pgrep -u $(whoami) python3 | xargs -r kill -9 2>/dev/null || true
  pgrep -u $(whoami) python | xargs -r kill -9 2>/dev/null || true
  pgrep -u $(whoami) vllm | xargs -r kill -9 2>/dev/null || true


  # wait until GPU memory usage smaller than 1GB
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
    sleep 1
  done

  # remove vllm config file
  rm -rf ~/.config/vllm

}

cleanup() {
  echo "Script interrupted, cleaning up..."
  kill_gpu_processes
  exit 130
}
trap cleanup INT TERM

server_command="VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
                --tensor-parallel-size 1 \
                --enable-chunked-prefill \
                --max-num-seqs 256 \
                --max-num-batched-tokens 256 \
                --disable-log-stats \
                --disable-log-requests \
                --swap-space 0"
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
result_filename="${OUTPUT_FOLDER}/output/vllm.json"
client_command="VLLM_USE_V1=1 python3 benchmark_vllm.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend vllm \
        --ignore-eos \
        --num-prompts 256 \
        --dataset-path ${kickstart_trace_file} \
        --save-result --save-detailed \
        --result-dir ../output/vllm \
        --result-filename ${result_filename}"

echo "Client command: $client_command"
bash -c "$client_command"
# clean up
kill -9 $server_pid
kill_gpu_processes

check_output_file $result_filename
echo -e "${GREEN}✓ vLLM test passed!${NC}"
echo ""
##################################################


############### Test LLAMA-Factory ###############
echo -e "${YELLOW}Testing LLAMA-Factory...${NC}"
cd "${LLAMA_FACTORY_FOLDER}"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/flexllm/kickstart.yaml
mkdir -p ${OUTPUT_FOLDER}/output/llama-factory
mv ./saves/* ${OUTPUT_FOLDER}/output/llama-factory/
check_output_file "${OUTPUT_FOLDER}/output/llama-factory/kickstart/lora/sft/train_results.json"
echo -e "${GREEN}✓ LLAMA-Factory test passed!${NC}"
echo ""
##################################################

# Final message + cleanup
echo -e "${GREEN}All kickstart tests passed successfully!${NC}"
echo "You can find the output files in the directory: ${OUTPUT_FOLDER}/output"
