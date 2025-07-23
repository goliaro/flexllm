#! /usr/bin/env bash
set -euo pipefail

cd "${BASH_SOURCE[0]%/*}/.."

cuda_version=12.4 python_version=3.12 ./flexflow-serve/docker/build.sh flexflow-environment

