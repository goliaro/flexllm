# flexllm


1. Run `cuda_version=12.4 python_version=3.12 ./docker/build.sh flexflow-environment`
2. Run `cuda_version=12.4 python_version=3.12 ./docker/run.sh flexflow-environment`
3. Run `git clone -b flexllm-aec --recursive git@github.com:goliaro/flexllm.git`
4. Run `cd flexllm; pip install -r requirements.txt`
5. Run `huggingface-cli login --token <HF TOKEN>`
6. `cd vllm; VLLM_USE_PRECOMPILED=1 pip install -e . --verbose`