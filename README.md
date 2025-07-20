# flexllm


1. Run `cuda_version=12.4 python_version=3.12 ./docker/build.sh flexflow-environment`
2. Run `cuda_version=12.4 python_version=3.12 ./docker/run.sh flexflow-environment`
3. Run `git clone -b flexllm-aec --recursive git@github.com:goliaro/flexllm.git`
4. Run `cd flexllm; pip install -r requirements.txt`
5. Run `huggingface-cli login --token <HF TOKEN>`
6. Run `./benchmarking/get_traces.sh`
7. Test: `vllm serve meta-llama/Llama-3.1-8B-Instruct`


If you run into the issue below:
```
root@<container id>:/flexllm# nvidia-smi
Failed to initialize NVML: Unknown Error
```
You will need to restart the container with `docker container restart <container id>`
