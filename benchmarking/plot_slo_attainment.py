from collections import defaultdict, namedtuple
import os, itertools, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field

def get_slo_attainment(df_original, tpot_slo_ms):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or decoding_step_idx is < 0
    df = df[(df["is_warmup_request"] == 0) & (df["decoding_step_idx"] >= 0)]
    group = df.groupby("request_guid", as_index=False)
    min_time = group["timestamp"].min()["timestamp"]
    max_time = group["timestamp"].max()["timestamp"]
    num_generated_tokens = group.size()["size"]
    tpots = (max_time - min_time) / num_generated_tokens / 1000

    below_threshold_percentage = (tpots < tpot_slo_ms).mean()
    return below_threshold_percentage

def get_throughput(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or decoding_step_idx is < 0
    df = df[(df["is_warmup_request"] == 0) & (df["decoding_step_idx"] >= 0)]
    # Exclude the last request to finish:
    # Identify the request_guid corresponding to the row with the latest timestamp
    last_request_guid = df.loc[df["timestamp"].idxmax()]["request_guid"]
    # Remove all rows with that request_guid
    df = df[df["request_guid"] != last_request_guid]
    # compute the throughput as the number of rows in (df) divided by the total time taken
    microsec_to_sec = 1_000_000
    total_time_sec = (df["timestamp"].max() - df["timestamp"].min()) / microsec_to_sec
    total_output_tokens = df.shape[0]
    return total_output_tokens / total_time_sec

def get_ft_throughput(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1 or request_step_idx is < 0
    df = df[df["is_warmup_step"] == 0]
    # compute the throughput as the number of rows in the filtered dataframe (df) divided by the total time taken
    microsec_to_sec = 1_000_000
    total_time_sec = (df["timestamp"].max() - df["timestamp"].min()) / microsec_to_sec
    total_output_tokens = df["num_finetuning_fwd_tokens"].sum()
    return total_output_tokens / total_time_sec


def get_slo_attainment_vllm(data, tpot_slo_ms):
    # data is assumed to be a dict with keys "itl" (a list of lists) and "output_lens" (a list of ints)
    # Compute the TPOT for each request: sum(itl[i]) divided by output_lens[i]
    tpots = []
    for itl, output_len in zip(data["itls"], data["output_lens"]):
        # Avoid division by zero if output_len is 0
        request_tpot = sum(itl) / output_len if output_len > 0 else np.nan
        tpots.append(request_tpot)

    tpots = np.array(tpots)
    # Calculate percentage of requests where TPOT is below the tpot_slo_ms threshold
    attainment_percentage = (tpots < tpot_slo_ms).mean()
    return attainment_percentage

def get_ttft(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    ttft = group.apply(lambda x: x[x["decoding_step_idx"] == 0]["timestamp"].values[0] - x[x["decoding_step_idx"] == -1]["timestamp"].values[0])/1000
    # convert to milliseconds from microseconds
    return ttft.mean().iloc[1], ttft.median().iloc[1], ttft.quantile(0.99).iloc[1]

def get_queueing_time(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    microsec_to_sec = 1_000_000
    # in each group, find the difference between the timestampt at request_step_idx=-1 and the timestamp at request_step_idx=-2.
    queueing_time = group.apply(lambda x: x[x["decoding_step_idx"] == -1]["timestamp"].values[0] - x[x["decoding_step_idx"] == -2]["timestamp"].values[0])/1000
    return queueing_time.mean().iloc[1], queueing_time.median().iloc[1], queueing_time.quantile(0.99).iloc[1]

@dataclass
class BenchmarkResult:
    directory: str = ""
    slo_attainments: dict = field(default_factory=lambda: defaultdict(list))
    inference_throughputs: dict = field(default_factory=lambda: defaultdict(list))
    finetuning_throughputs: dict = field(default_factory=lambda: defaultdict(list))
    queueing_times: dict = field(default_factory=lambda: defaultdict(list))
    ttfts: dict = field(default_factory=lambda: defaultdict(list))




models=["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
llama_factory_model_names=["t1_llama_8B/lora/sft", "t1_qwen_14B/lora/sft", "t1_qwen_32B/lora/sft"]
tp_degrees=[1, 2, 4]
kv_cache_slots_values=[70000, 70000, 60000]
qps_values=[1.0, 2.0, 3.0, 4.0, 5.0]
tpot_slos_ms=[45,75,75]
data={
    "coserving": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/coserving/profiling"),
    "spatial-sharing": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/spatial_sharing/profiling"),
    "temporal-sharing": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing/profiling"),
    "spatial-sharing-limited": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/spatial_sharing_limited/profiling"),
    "temporal-sharing-limited": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing_limited/profiling"),
    "vllm-50pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/vllm"),
    "vllm-25pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/vllm"),
    "llama-factory-50pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/LLaMA-Factory/saves"),
    "llama-factory-25pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/LLaMA-Factory/saves"),
}

# gather non-vllm data
for experiment_type in data.keys():
    for i, model in enumerate(models):
        model_ = model.replace("/", "_").lower()
        tp_degree = tp_degrees[i]
        kv_cache_slots = kv_cache_slots_values[i]
        tpot_slo_ms=tpot_slos_ms[i]
        data[experiment_type].slo_attainments[model_] = []
        data[experiment_type].inference_throughputs[model_] = []
        data[experiment_type].finetuning_throughputs[model_] = []
        data[experiment_type].queueing_times[model_] = []
        data[experiment_type].ttfts[model_] = []
        for qps in qps_values:
            if "vllm" not in experiment_type and "llama-factory" not in experiment_type:
                filepath= os.path.join(data[experiment_type].directory, f"inference_request_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_0.csv")
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    slo_attainment_pct = get_slo_attainment(df, tpot_slo_ms)
                    data[experiment_type].slo_attainments[model_].append(slo_attainment_pct)
                    data[experiment_type].inference_throughputs[model_].append(get_throughput(df))
                    data[experiment_type].queueing_times[model_].append(get_queueing_time(df)[0])
                    data[experiment_type].ttfts[model_].append(get_ttft(df)[0])
                else:
                    print(f"File {filepath} does not exist.")
                    data[experiment_type].slo_attainments[model_].append(np.nan)
                    data[experiment_type].inference_throughputs[model_].append(np.nan)
                    data[experiment_type].queueing_times[model_].append(np.nan)
                    data[experiment_type].ttfts[model_].append(np.nan)
                
                step_filepath=os.path.join(data[experiment_type].directory, f"step_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_0.csv")
                if os.path.exists(step_filepath):
                    df = pd.read_csv(step_filepath)
                    ft_throughput = get_ft_throughput(df)
                    data[experiment_type].finetuning_throughputs[model_].append(ft_throughput)
                else:
                    print(f"File {step_filepath} does not exist.")
                    data[experiment_type].finetuning_throughputs[model_].append(np.nan)
            elif "vllm" in experiment_type:
                vllm_qps = qps*2 if "vllm-50pct" in experiment_type else qps*4
                filepath= os.path.join(data[experiment_type].directory, f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps.json")
                if not os.path.exists(filepath):
                    filepath = os.path.join(data[experiment_type].directory, f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps_.json")
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        vllm_data = json.load(f)
                    slo_attainment_pct = get_slo_attainment_vllm(vllm_data, tpot_slo_ms)
                    denominator = 2 if "50pct" in experiment_type else 4
                    inference_throughput = vllm_data["output_throughput"] / denominator
                    data[experiment_type].ttfts[model_].append(vllm_data["mean_ttft_ms"])
                else:
                    print(f"File {filepath} does not exist.")
                    slo_attainment_pct = np.nan
                    inference_throughput = np.nan
                    data[experiment_type].ttfts[model_].append(np.nan)
                data[experiment_type].slo_attainments[model_].append(slo_attainment_pct)
                data[experiment_type].inference_throughputs[model_].append(inference_throughput)
                data[experiment_type].finetuning_throughputs[model_].append(np.nan)
                data[experiment_type].queueing_times[model_].append(0)
            elif "llama-factory" in experiment_type:
                llama_factory_model_name = llama_factory_model_names[i]
                filepath = os.path.join(data[experiment_type].directory, llama_factory_model_name, f"train_results.json")
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        llama_factory_data = json.load(f)
                    denominator = 2 if "50pct" in experiment_type else 4
                    finetuning_throughput = llama_factory_data["train_tokens_per_second"] / denominator
                else:
                    print(f"File {filepath} does not exist.")
                    finetuning_throughput = np.nan
                data[experiment_type].slo_attainments[model_].append(np.nan)
                data[experiment_type].inference_throughputs[model_].append(np.nan)
                data[experiment_type].finetuning_throughputs[model_].append(finetuning_throughput)
                data[experiment_type].queueing_times[model_].append(np.nan)
                data[experiment_type].ttfts[model_].append(np.nan)

experiment_types = [et for et in data.keys()]
# Convert QPS values to floats for plotting
arrival_rates = [float(q) for q in qps_values]
# Get processed model names matching the keys in our data
model_keys = [m.replace("/", "_").lower() for m in models]

output_folder="/global/homes/g/goliaro/flexllm/benchmarking/plots/e2e"
os.makedirs(output_folder, exist_ok=True)

##########################################################################
################### TPOT SLO ATTAINMENT PLOT #############################
##########################################################################
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, model_key in enumerate(model_keys):
    model_name = models[i]
    ax = axs[i]
    for exp in experiment_types:
        if "llama-factory" in exp:
            continue
        # Plot each experiment type's SLO attainment vs arrival rate
        slo_values = data[exp].slo_attainments[model_key]
        ax.plot(arrival_rates, slo_values, marker='o', label=exp)
    ax.set_xticks(arrival_rates)
    ax.set_title(f"SLO Attainment (TPOT <{tpot_slos_ms[i]} ms)\n{model_name} (TP={tp_degrees[i]})")
    ax.set_xlabel("Arrival Rate (QPS)")
    ax.set_ylabel("SLO Attainment (%)")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "slo_attainment.pdf"), dpi=300)

##########################################################################
################### FINETUNING THROUGHPUT PLOT ###########################
##########################################################################
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, model_key in enumerate(model_keys):
    model_name=models[i]
    ax = axs[i]
    for exp in experiment_types:
        if "vllm" in exp:
            continue
        # Plot each experiment type's finetuning throughput vs arrival rate
        throughput_values = data[exp].finetuning_throughputs[model_key]
        ax.plot(arrival_rates, throughput_values, marker='o', label=exp)
    # ax.set_title(model_key)
    ax.set_xticks(arrival_rates)
    ax.set_title(f"Finetuning Throughput\n{model_name} (TP={tp_degrees[i]})")
    ax.set_xlabel("Arrival Rate (QPS)")
    ax.set_ylabel("Finetuning Throughput (tokens/sec)")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "finetuning_throughput.pdf"), dpi=300)

##########################################################################
################### INFERENCE THROUGHPUT PLOT ###########################
##########################################################################
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, model_key in enumerate(model_keys):
    model_name=models[i]
    ax = axs[i]
    for exp in experiment_types:
        if "llama-factory" in exp:
            continue
        # Plot each experiment type's finetuning throughput vs arrival rate
        throughput_values = data[exp].inference_throughputs[model_key]
        ax.plot(arrival_rates, throughput_values, marker='o', label=exp)
    ax.set_xticks(arrival_rates)
    ax.set_title(f"Inference Throughput\n{model_name} (TP={tp_degrees[i]})")
    ax.set_xlabel("Arrival Rate (QPS)")
    ax.set_ylabel("Inference Throughput (tokens/sec)")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "inference_throughput.pdf"), dpi=300)

##########################################################################
###################### QUEUE + TTFT PLOT #################################
##########################################################################
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, model_key in enumerate(model_keys):
    model_name=models[i]
    ax = axs[i]
    for exp in experiment_types:
        if "llama-factory" in exp:
            continue
        # Plot each experiment type's finetuning throughput vs arrival rate
        # print(model_name)
        # print(exp)
        # print(data[exp].queueing_times[model_key])
        # print(data[exp].ttfts[model_key])
        y_values = [(a+b)/1000 for (a,b) in zip(data[exp].queueing_times[model_key], data[exp].ttfts[model_key])]
        # print(arrival_rates)
        # print(y_values)
        # print()
        ax.plot(arrival_rates, y_values, marker='o', label=exp)
    ax.set_xticks(arrival_rates)
    ax.set_title(f"Queueing time + TTFT\n{model_name} (TP={tp_degrees[i]})")
    ax.set_xlabel("Arrival Rate (QPS)")
    ax.set_ylabel("Queueing time + TTFT (s)")
    ax.set_ylim(0, 60)
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "ttft.pdf"), dpi=300)