#! /usr/bin/env bash
set -euo pipefail

cd "${BASH_SOURCE[0]%/*}/.."

docker run -d --name flexllm --gpus all --shm-size=8192m --cap-add=SYS_PTRACE --entrypoint="" flexflow-environment-cuda-12.4:latest sleep infinity
