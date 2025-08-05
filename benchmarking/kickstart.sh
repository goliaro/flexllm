#!/bin/bash

cd "${BASH_SOURCE[0]%/*}/.."
TRACES_FOLDER="./traces"
OUTPUT_FOLDER="./output/kickstart"

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
        read -p "Please confirm (y/yes/Y/YES to continue, anything else to exit): " response
        
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

echo -e "${GREEN}=== FlexLLM Setup Verification ===${NC}"
echo ""
echo "This script will verify that you have completed all the preparation steps."
echo "Please answer 'y', 'yes', 'Y', or 'YES' to confirm each step."
echo ""

# Step 1: Build container
get_confirmation "Have you run './docker/build_container.sh' to build the Docker container?"

# Step 2: Start container
get_confirmation "Have you run './docker/start_container.sh' to start the container (it should be running in the background)?"

# Step 3: Setup FlexLLM
echo ""
echo -e "${YELLOW}Step 3 involves installing libraries and downloading Huggingface models.${NC}"
echo "This step requires a Huggingface token to access the following models:"
echo "  - meta-llama/Llama-3.1-8B-Instruct"
echo "  - Qwen/Qwen2.5-14B-Instruct" 
echo "  - Qwen/Qwen2.5-32B-Instruct"
echo ""
get_confirmation "Have you run './docker/setup_flexllm.sh' and provided your Huggingface token when prompted?"

# Step 4: Attach to container
get_confirmation "Have you run './docker/attach_to_container.sh' to open a terminal connected to the container?"

echo ""
echo -e "${GREEN}✓ All preparation steps have been confirmed!${NC}"
echo -e "${GREEN}You should now be ready to run FlexLLM experiments.${NC}"
echo ""

# Check that the script is run from within docker
if [ ! -f /.dockerenv ]; then
    echo -e "${RED}Error: Please run this script from within the Docker container.${NC}"
    exit 1
fi

# Check that the gpus are available
gpu_count=$(nvidia-smi -L | wc -l)
if [ "$gpu_count" -lt 4 ]; then
    echo -e "${RED}Error: At least 4 NVIDIA GPUs are required. Found $gpu_count.${NC}"
    exit 1
fi

# Check that the traces are available
if [ ! -d $TRACES_FOLDER ]; then
    echo -e "${RED}Error: Traces directory not found. Please run get_traces.sh first.${NC}"
    exit 1
fi

# check that traces actually exist



#################### Test co-serving ####################
cd "${BASH_SOURCE[0]%/*}/../flexflow-serve/build"
source ./set_python_envs.sh

mkdir -p ../../kickstart/output/profiling

./inference/flexllm/peft_train \
    -ll:cpu 16 -ll:gpu 1 -ll:util 16 \
    -ll:fsize 77000 -ll:zsize 40000 -ll:csize 4096 \
    -llm-model meta-llama/Llama-3.1-8B-Instruct --fusion \
    -tensor-parallelism-degree 1 \
    -prompt ../../kickstart/inference_trace.json \
    -peft-model kickstart-lora --peft-support-mode COSERVING \
    -finetuning-dataset ../../kickstart/finetuning_trace.json \
    --max-training-epochs 10000 \
    --gradient-accumulation-steps 8 \
    --num-layers-per-finetuning-step 2 \
    --num-logging-steps 100 \
    -output-file ../../kickstart/output/kickstart.json \
    -profiling-folder ../../kickstart/output/profiling \
    --max-requests-per-batch 256 \
    --max-tokens-per-batch 256 \
    --max-sequence-length 8192 \
    --num-kv-cache-slots 70000 \
    --ignore-eos --warmup --log-instance-creation \
    2>&1 | tee ../../kickstart/output/kickstart.log
##################################################

#################### Test vLLM ####################
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
  lsof -t -i:8000 | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  pgrep python | xargs -r kill -9
  pgrep vllm | xargs -r kill -9


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

cd "${BASH_SOURCE[0]%/*}"

##################################################


############### Test LLAMA-Factory ###############
cd "${BASH_SOURCE[0]%/*}"

##################################################
