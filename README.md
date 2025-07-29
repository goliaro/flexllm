# FlexLLM (paper 183 @ NSDI 2026) - Artifact Evaluation

Paper PDF: [nsdi26spring-paper183.pdf](./nsdi26spring-paper183.pdf)

## Hardware setup
To begin, please spin up a machine with the following characteristics:
- 4 NVIDIA A100-SXM4-80GB GPUs
- CUDA 12.4
- Docker support with NVIDIA container runtime
- 500GB+ disk memory
If you are using AWS, please create a `p4de.24xlarge` instance with the `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)` AMI. If you do not have access to such a machine, let us know and we will start a machine for you.

Once you have started the machine (or we started one for you), please connect to the machine via SSH. 

## Preparation
To start, please download the code with `git clone --recursive https://github.com/goliaro/flexllm.git`. Then, follow the steps below to build a Docker container where you will be able to run all experiments.

1. Run `./docker/build_container.sh` to build the container
2. Run `./docker/start_container.sh` to start the container. It will continue running in the background until stopped.
3. Run `./docker/setup_flexllm.sh` to install all the required libraries and download the Huggingface models in the container. When prompted, please provide your huggingface token to access the following models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`. If you do not have a token, we can provide one.
4. Run `./docker/attach_to_container.sh` to open a new terminal connected to the container. You can run this multiple times if you'd like to connect multiple terminal windows.


## Running the experiments
To run all the experiments, launch the commands below one at a time within the Docker container. The commands use `nohup` to ensure that they will keep running if the SSH connection is broken. After launching a command, you can feel free to disconnect and come back later to check the progression. To check the progress (in real-time), you can run `tail -f <output_file>` (replace with `output1.log`, `output2.log` or `output3.log`) from the `/` folder in the container. If you are using a `tmux` terminal, you should avoid using `nohup` and instead use the tmux regular functionalities to run each script and check the output. 

You can tell that each experiment has done when no additional output is being appended to the output file. In the last few lines, you should also be able to see a message that says: "All experiments completed!" or something similar. Before launching the next experiment, to be safe, please also check that the GPU memory utilization is at 0% (by running `nvidia-smi`).
1. Run the separate baseline experiments (LLaMAFactory + vLLM) with: `nohup bash -c './flexllm/benchmarking/finetuning/benchmark_llamafactory.sh && ./flexllm/benchmarking/vllm_online/run.sh' > output1.log 2>&1 &`
2. Run the spatial/temporal baseline experiments with: `nohup bash -c './flexllm/benchmarking/coserving/run_spatial_sharing.sh && ./flexllm/benchmarking/coserving/run_temporal_sharing.sh' > output2.log 2>&1 &`
3. Run the co-serving experiments with: `nohup ./flexllm/benchmarking/coserving/run_coserving.sh > output3.log 2>&1 &`

Once you are done with all experiments, you can run `./flexllm/benchmarking/coserving/check_all_experiments.sh` to ensure that all FlexLLM experiments have produced their expected output successfully. If some data point is missing for any reason, you can rerun the missing experiments by calling the corresponding script again. The script will automatically detect which results are missing and only rerun the corresponding tasks. For example:
- If some spatial sharing outputs are missing, run `./flexllm/benchmarking/coserving/run_spatial_sharing.sh`
- If some temporal sharing outputs are missing, run `./flexllm/benchmarking/coserving/run_temporal_sharing.sh`
- If some co-serving outputs are missing, run `./flexllm/benchmarking/coserving/run_coserving.sh`
You do not have to rerun with `nohup`, but you can if you prefer.


## Parsing and plotting the results
After all experiments have completed, you should run the `./flexllm/benchmarking/parse_data.py` script to parse all the output data into a single pickle file. This will take about 10mins to complete. The output file will be saved at `./flexllm/benchmarking/output/benchmark_data.pkl`. Note that if that file already exists, you will need to delete it before it can be overwritten. After creating the pickle file, you can run `./flexllm/benchmarking/plot_data.py` to plot the results. The script will produce two plots: `./flexllm/benchmarking/output/external_baselines.pdf` (Fig 10 in the paper) and `./flexllm/benchmarking/output/internal_baselines.pdf` (Fig 11 in the paper).

## Downloading the plots and results to the host
To download the plots to the host, `cd` to the desired directory (on the host) where you'd like to save the output, and run the following commands:

```
docker cp <container id>:/flexllm/benchmarking/output/external_baselines.pdf ./
docker cp <container id>:/flexllm/benchmarking/output/internal_baselines.pdf ./
```

Replace <container id> with the ID of your docker container. You can obtain this ID by running `docker ps`. The output will look similar to this (in the example below, the container id is `0cbc018ce9e5`)
```
ubuntu@ip-172-31-38-174:~$ docker ps
CONTAINER ID   IMAGE                                   COMMAND            CREATED      STATUS          PORTS     NAMES
0cbc018ce9e5   flexflow-environment-cuda-12.4:latest   "sleep infinity"   6 days ago   Up 43 minutes             flexllm
```

We also recommend that you download the final output results by zipping the contents of the output folder, and downloading to the host in a similar fashion:

- On the docker machine, run: `zip -r output_results.zip ./flexllm/benchmarking/output`
- On the host machine, run: `docker cp <container id>:/output_results.zip ./`

You can also use these instructions above to periodically checkpoint the output results before being done with all experiments. If you choose to do so, ensure that you are using different names for your zip archive.

## Teardown
⚠️⚠️⚠️ Make sure to save the plots and/or results before proceeding. The step below cannot be undone. ⚠️⚠️⚠️

- Run `./docker/cleanup_containers.sh` after you are done with the experiments to stop and destroy the container and all docker images/data.

## Troubleshooting
If you are on AWS and your container cannot find the GPUs (a well-known issue):
```
root@<container id>:/flexllm# nvidia-smi
Failed to initialize NVML: Unknown Error
```
Use this fix (on the host machine, not within docker): https://stackoverflow.com/a/78137688

If that still doesn't help, please restart the container with `docker container restart <container id>`
