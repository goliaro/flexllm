#! /usr/bin/env bash
set -euo pipefail

cd "${BASH_SOURCE[0]%/*}/.."

docker exec -it flexllm /bin/bash
