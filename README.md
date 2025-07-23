# flexllm


## Preparation
1. Run `./docker/build_container.sh`
2. Run `./docker/start_container.sh`
3. Run `./docker/setup_flexllm.sh`, providing your huggingface token when requested

4. Run `./docker/attach_to_container.sh`
5. Run `./docker/cleanup_containers.sh` after you are done

## Experiments
1. (~9h) Run the separate baseline experiments (LLaMAFactory + vLLM) with: `nohup bash -c './flexllm/benchmarking/finetuning/benchmark_llamafactory.sh && ./flexllm/benchmarking/vllm_online/run.sh' > output.log 2>&1 &`
2. Run the spatial/temporal baseline experiments with: `nohup bash -c './flexllm/benchmarking/coserving/run_spatial_sharing.sh && ./flexllm/benchmarking/coserving/run_temporal_sharing.sh' > output.log 2>&1 &`

## Troubleshooting
If you run into the issue below:
```
root@<container id>:/flexllm# nvidia-smi
Failed to initialize NVML: Unknown Error
```
Use this fix (on the host machine, not within docker): https://stackoverflow.com/a/78137688

Or the following temporary fix: restart the container with `docker container restart <container id>`
