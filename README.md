# flexllm


## Preparation
1. Run `cuda_version=12.4 python_version=3.12 ./docker/build.sh flexflow-environment`
2. Run `cuda_version=12.4 python_version=3.12 ./docker/run.sh flexflow-environment`
3. Run `git clone -b flexllm-aec --recursive git@github.com:goliaro/flexllm.git`
4. Run `cd flexllm; pip install -r requirements.txt`
5. Run `huggingface-cli login --token <HF TOKEN>`
6. Run `./benchmarking/get_traces.sh`
7. Run `cd flexflow-serve && mkdir build && cd build && ../config/config.linux && make -j && cd ..`
8. Run `cd build && source set_python_envs.sh && cd .. && python inference/utils/download_hf_model.py --half-precision-only meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct`

## Experiments
1. (~9h) Run the baseline experiments (LLaMAFactory + vLLM) with: `nohup bash -c './flexllm/benchmarking/finetuning/benchmark_llamafactory.sh && ./flexllm/benchmarking/vllm_online/run.sh' > output.log 2>&1 &`
2. 

## Troubleshooting
If you run into the issue below:
```
root@<container id>:/flexllm# nvidia-smi
Failed to initialize NVML: Unknown Error
```
Use this fix (on the host machine, not within docker): https://stackoverflow.com/a/78137688

Or the following temporary fix: restart the container with `docker container restart <container id>`
