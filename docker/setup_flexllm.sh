#! /usr/bin/env bash
set -euo pipefail

cd "${BASH_SOURCE[0]%/*}/.."

read -sp "Enter your Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN

docker exec -it -e HF_TOKEN flexllm /bin/bash -c '
set -x
set -e

# Clone the codebase inside the container
git clone -b flexllm-aec --recursive https://github.com/goliaro/flexllm.git

# Install the Python packages that are needed for the experiments
cd flexllm
pip install -r requirements.txt

# Login to Hugging Face, and download the traces for the experiments
huggingface-cli login --token $HF_TOKEN
./benchmarking/get_traces.sh

# Build FlexLLM
cd flexflow-serve 
mkdir -p build 
cd build
../config/config.linux 
make -j

# Download the huggingface models to be used in the experiments
source ./set_python_envs.sh 
cd .. 
python inference/utils/download_hf_model.py --half-precision-only meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct
'
