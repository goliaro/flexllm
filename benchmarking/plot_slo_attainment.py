from collections import defaultdict, namedtuple
import os, itertools, json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from dataclasses import dataclass, field
import pickle

def get_tpot_slo_attainment(df_original, tpot_slo_ms):
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


def get_tpot_slo_attainment_vllm(data, tpot_slo_ms):
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

def get_slo_attainment(df_original, tpot_slo_ms, ttft_slo_ms):
    df = df_original.copy()
    # Only consider non-warmup requests
    df = df[df["is_warmup_request"] == 0]
    groups = df.groupby("request_guid")
    valid_requests = []
    for _, group in groups:
        # Compute TPOT only on rows with a non-negative decoding_step_idx
        group_valid = group[group["decoding_step_idx"] >= 0]
        if group_valid.empty:
            continue
        num_tokens = group_valid.shape[0]
        tpot = (group_valid["timestamp"].max() - group_valid["timestamp"].min()) / num_tokens / 1000.0

        # compute ttft and queueing delay
        if not (group["decoding_step_idx"] == -2).any() or not (group["decoding_step_idx"] == 0).any():
            continue
        try:
            total_time = (group[group["decoding_step_idx"] == 0].iloc[0]["timestamp"] -
                  group[group["decoding_step_idx"] == -2].iloc[0]["timestamp"]) / 1000.0
        except IndexError:
            continue
        valid_requests.append((tpot, total_time))

    if not valid_requests:
        return np.nan

    valid_requests = np.array(valid_requests)
    # Check the SLO conditions for each request
    meets_slo = (valid_requests[:, 0] < tpot_slo_ms) & (valid_requests[:, 1] < ttft_slo_ms)
    return meets_slo.mean()


def get_slo_attainment_vllm(data, tpot_slo_ms, ttft_slo_ms):
    # data is assumed to be a dict with keys:
    # "itls": a list of lists of inter-token latencies for each request,
    # "output_lens": a list of ints representing the output length for each request,
    # "ttft": a list of floats representing the time to first token (in ms) for each request.
    count = 0
    total = 0
    for itl, output_len, ttft in zip(data["itls"], data["output_lens"], data["ttfts"]):
        if output_len <= 0:
            continue  # Skip requests with invalid output length
        total += 1
        tpot = sum(itl) / output_len
        if (tpot < tpot_slo_ms) and (ttft*1000 < ttft_slo_ms):
            count += 1
    return count / total if total > 0 else np.nan

@dataclass
class BenchmarkResult:
    directory: str = ""
    tpot_slo_attainments: dict = field(default_factory=lambda: defaultdict(list))
    inference_throughputs: dict = field(default_factory=lambda: defaultdict(list))
    finetuning_throughputs: dict = field(default_factory=lambda: defaultdict(list))
    queueing_times: dict = field(default_factory=lambda: defaultdict(list))
    ttfts: dict = field(default_factory=lambda: defaultdict(list))
    slo_attainments: dict = field(default_factory=lambda: defaultdict(list))




models=["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
llama_factory_model_names=["t1_llama_8B/lora/sft", "t1_qwen_14B/lora/sft", "t1_qwen_32B/lora/sft"]
tp_degrees=[1, 2, 4]
kv_cache_slots_values=[70000, 70000, 60000]
qps_values=[1.0, 2.0, 3.0, 4.0, 5.0]
tpot_slos_ms=[45,75,75]
ttft_slos_ms=[5000, 5000, 5000]

output_folder="/global/homes/g/goliaro/flexllm/benchmarking/plots/e2e"
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, "benchmark_data.pkl")

if not os.path.exists(save_path):
    data={
        "coserving": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/coserving/profiling"),
        "spatial-sharing": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/spatial_sharing/profiling"),
        "spatial-sharing-limited": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/spatial_sharing_limited/profiling"),
        # "temporal-sharing-1": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing/profiling"),
        "temporal-sharing-64": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing/64/profiling"),
        "temporal-sharing-128": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing/128/profiling"),
        "temporal-sharing-256": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing/256/profiling"),
        "temporal-sharing-512": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing/512/profiling"),
        # "temporal-sharing-limited": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/temporal_sharing_limited/profiling"),
        "vllm-25pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/vllm"),
        "vllm-50pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/vllm"),
        "vllm-75pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/benchmarking/output/vllm"),
        "llama-factory-25pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/LLaMA-Factory/saves"),
        "llama-factory-50pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/LLaMA-Factory/saves"),
        "llama-factory-75pct": BenchmarkResult(directory="/global/homes/g/goliaro/flexllm/LLaMA-Factory/saves"),
    }

    # gather non-vllm data
    for experiment_type in data.keys():
        for i, model in enumerate(models):
            model_ = model.replace("/", "_").lower()
            tp_degree = tp_degrees[i]
            kv_cache_slots = kv_cache_slots_values[i]
            tpot_slo_ms=tpot_slos_ms[i]
            ttft_slo_ms=ttft_slos_ms[i]
            data[experiment_type].tpot_slo_attainments[model_] = []
            data[experiment_type].inference_throughputs[model_] = []
            data[experiment_type].finetuning_throughputs[model_] = []
            data[experiment_type].queueing_times[model_] = []
            data[experiment_type].ttfts[model_] = []
            data[experiment_type].slo_attainments[model_] = []
            for qps in qps_values:
                if "vllm" not in experiment_type and "llama-factory" not in experiment_type:
                    filepath= os.path.join(data[experiment_type].directory, f"inference_request_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_0.csv")
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath)
                        data[experiment_type].tpot_slo_attainments[model_].append(get_tpot_slo_attainment(df, tpot_slo_ms))
                        data[experiment_type].inference_throughputs[model_].append(get_throughput(df))
                        data[experiment_type].queueing_times[model_].append(get_queueing_time(df)[0])
                        data[experiment_type].ttfts[model_].append(get_ttft(df)[0])
                        data[experiment_type].slo_attainments[model_].append(get_slo_attainment(df, tpot_slo_ms, ttft_slo_ms))
                    else:
                        print(f"File {filepath} does not exist.")
                        data[experiment_type].tpot_slo_attainments[model_].append(np.nan)
                        data[experiment_type].inference_throughputs[model_].append(np.nan)
                        data[experiment_type].queueing_times[model_].append(np.nan)
                        data[experiment_type].ttfts[model_].append(np.nan)
                        data[experiment_type].slo_attainments[model_].append(np.nan)
                    
                    step_filepath=os.path.join(data[experiment_type].directory, f"step_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_0.csv")
                    if os.path.exists(step_filepath):
                        df = pd.read_csv(step_filepath)
                        ft_throughput = get_ft_throughput(df)
                        data[experiment_type].finetuning_throughputs[model_].append(ft_throughput)
                    else:
                        print(f"File {step_filepath} does not exist.")
                        data[experiment_type].finetuning_throughputs[model_].append(np.nan)
                elif "vllm" in experiment_type:
                    vllm_qps = 0
                    denominator = 0
                    if "vllm-25pct" in experiment_type:
                        vllm_qps = round(qps*4, 1)
                        denominator = 4
                    elif "vllm-50pct" in experiment_type:
                        vllm_qps = round(qps*2, 1)
                        denominator = 2
                    elif "vllm-75pct" in experiment_type:
                        vllm_qps = round(qps*4/3, 1)
                        denominator = 4/3
                    filepath= os.path.join(data[experiment_type].directory, f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps.json")
                    if not os.path.exists(filepath):
                        filepath = os.path.join(data[experiment_type].directory, f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps_.json")
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            vllm_data = json.load(f)
                        tpot_slo_attainment_pct = get_tpot_slo_attainment_vllm(vllm_data, tpot_slo_ms)
                        inference_throughput = vllm_data["output_throughput"] / denominator
                        data[experiment_type].ttfts[model_].append(vllm_data["mean_ttft_ms"])
                        slo_attainment_pct = get_slo_attainment_vllm(vllm_data, tpot_slo_ms, ttft_slo_ms)
                    else:
                        print(f"File {filepath} does not exist.")
                        tpot_slo_attainment_pct = np.nan
                        inference_throughput = np.nan
                        data[experiment_type].ttfts[model_].append(np.nan)
                        slo_attainment_pct = np.nan
                    data[experiment_type].tpot_slo_attainments[model_].append(tpot_slo_attainment_pct)
                    data[experiment_type].inference_throughputs[model_].append(inference_throughput)
                    data[experiment_type].finetuning_throughputs[model_].append(np.nan)
                    data[experiment_type].queueing_times[model_].append(0)
                    data[experiment_type].slo_attainments[model_].append(slo_attainment_pct)
                elif "llama-factory" in experiment_type:
                    llama_factory_model_name = llama_factory_model_names[i]
                    filepath = os.path.join(data[experiment_type].directory, llama_factory_model_name, f"train_results.json")
                    denominator = 0
                    if "25pct" in experiment_type:
                        denominator = 4
                    elif "50pct" in experiment_type:
                        denominator = 2
                    elif "75pct" in experiment_type:
                        denominator = 4/3
                    
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            llama_factory_data = json.load(f)
                        finetuning_throughput = llama_factory_data["train_tokens_per_second"] / denominator
                    else:
                        print(f"File {filepath} does not exist.")
                        finetuning_throughput = np.nan
                    data[experiment_type].tpot_slo_attainments[model_].append(np.nan)
                    data[experiment_type].inference_throughputs[model_].append(np.nan)
                    data[experiment_type].finetuning_throughputs[model_].append(finetuning_throughput)
                    data[experiment_type].queueing_times[model_].append(np.nan)
                    data[experiment_type].ttfts[model_].append(np.nan)
                    data[experiment_type].slo_attainments[model_].append(np.nan)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Benchmark data saved to {save_path}")
else:
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    print(f"Benchmark data loaded from {save_path}")

experiment_types = [et for et in data.keys()]
# Convert QPS values to floats for plotting
arrival_rates = [4*float(q) for q in qps_values]
# Get processed model names matching the keys in our data
model_keys = [m.replace("/", "_").lower() for m in models]


##########################################################################
################### TPOT SLO ATTAINMENT PLOT #############################
##########################################################################
if False:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, model_key in enumerate(model_keys):
        model_name = models[i]
        ax = axs[i]
        for exp in experiment_types:
            if "llama-factory" in exp:
                continue
            # Plot each experiment type's SLO attainment vs arrival rate
            slo_values = data[exp].tpot_slo_attainments[model_key]
            ax.plot(arrival_rates, slo_values, marker='o', label=exp)
        ax.set_xticks(arrival_rates)
        ax.set_title(f"TPOT SLO Attainment (TPOT <{tpot_slos_ms[i]} ms)\n{model_name} (TP={tp_degrees[i]})")
        ax.set_xlabel("Arrival Rate (QPS)")
        ax.set_ylabel("TPOT SLO Attainment (%)")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "tpot_slo_attainment.pdf"), dpi=300)

##########################################################################
################### FINETUNING THROUGHPUT PLOT ###########################
##########################################################################
if False:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, model_key in enumerate(model_keys):
        model_name=models[i]
        ax = axs[i]
        for exp in experiment_types:
            if "vllm" in exp or exp=="temporal-sharing-1" or exp=="spatial-sharing":
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
if False:
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
if False:
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

##########################################################################
###################### SLO ATTAINMENT PLOT ###############################
##########################################################################
if False:
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
        ax.set_title(f"SLO Attainment (TPOT <{tpot_slos_ms[i]} ms, TTFT <{ttft_slos_ms[i]/1000} s)\n{model_name} (TP={tp_degrees[i]})")
        ax.set_xlabel("Arrival Rate (QPS)")
        ax.set_ylabel("SLO Attainment (%)")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "slo_attainment.pdf"), dpi=300)

##########################################################################
#########################    E2E PLOT    #################################
##########################################################################
fig, axs = plt.subplots(3, 3, figsize=(12, 7))
model_names_simplified = ["Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"]
markers = ['o', 's', '^', 'D']  # Different dot types for the 4 curves

for j, model_key in enumerate(model_keys):
    model_name = model_names_simplified[j]
    # Only add the column title once at the top of each column
    axs[0, j].set_title(f"{model_name} (TP={tp_degrees[j]})", fontsize=12)
    
    # Row 0: SLO Attainment
    ax = axs[0, j]
    for idx, exp in enumerate(["coserving", "vllm-25pct", "vllm-50pct", "vllm-75pct"]):
        slo_values = data[exp].slo_attainments[model_key]
        ax.plot(arrival_rates, slo_values, marker=markers[idx], label=exp if j == 0 else None)
    ax.set_xticks(arrival_rates)
    ax.set_ylim(0, 1)
    if j == 0:
        ax.set_ylabel("\nSLO Attainment (%)", fontweight='bold')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # Row 1: Finetuning Throughput
    ax = axs[1, j]
    for idx, exp in enumerate(["coserving", "llama-factory-75pct", "llama-factory-50pct", "llama-factory-25pct"]):
        throughput_values = [4*x for x in data[exp].finetuning_throughputs[model_key]]
        # print(f"Finetuning throughput - {exp} - {model_key}: {throughput_values}")
        ax.plot(arrival_rates, throughput_values, marker=markers[idx], label=exp if j == 0 else None)
    ax.set_xticks(arrival_rates)
    ax.set_ylim(0)
    if j == 0:
        ax.set_ylabel("Finetuning\nThroughput\n(tokens/sec)", fontweight='bold')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # Row 2: Inference Throughput
    ax = axs[2, j]
    for idx, exp in enumerate(["coserving", "vllm-25pct", "vllm-50pct", "vllm-75pct"]):
        throughput_values = [4*x for x in data[exp].inference_throughputs[model_key]]
        ax.plot(arrival_rates, throughput_values, marker=markers[idx], label=exp if j == 0 else None)
    ax.set_xticks(arrival_rates)
    ax.set_xlabel("Arrival Rate (req/s)", fontweight='bold')
    ax.set_ylim(0)
    if j == 0:
        ax.set_ylabel("Inference\nThroughput\n(tokens/sec)", fontweight='bold')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# Add one common legend below the whole plot using handles from the first column
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, ["Collie", "Separate cluster (25% vLLM - 75% LLaMA-Factory)", "Separate cluster (50% vLLM - 50% LLaMA-Factory)", "Separate cluster (75% vLLM - 25% LLaMA-Factory)"], loc="upper center", fontsize=12, ncol=2 )
plt.savefig(os.path.join(output_folder, "combined_plot.pdf"), dpi=300, bbox_inches='tight')

##########################################################################
############ E2E comparison with Spatial Sharing PLOT    #################
##########################################################################
if False:
    fig, axs = plt.subplots(3, 3, figsize=(12, 7))
    model_names_simplified = ["Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"]
    markers = ['o', 's', '^', 'D']  # Different dot types for the 4 curves

    for j, model_key in enumerate(model_keys):
        model_name = model_names_simplified[j]
        # Only add the column title once at the top of each column
        axs[0, j].set_title(f"{model_name} (TP={tp_degrees[j]})", fontsize=12)
        
        # Row 0: SLO Attainment
        ax = axs[0, j]
        for idx, exp in enumerate(["coserving", "spatial-sharing", "spatial-sharing-limited"]):
            slo_values = data[exp].slo_attainments[model_key]
            ax.plot(arrival_rates, slo_values, marker=markers[idx], label=exp if j == 0 else None)
        ax.set_xticks(arrival_rates)
        ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel("\nSLO Attainment (%)")
        ax.grid(True)
        
        # Row 1: Finetuning Throughput
        ax = axs[1, j]
        for idx, exp in enumerate(["coserving", "spatial-sharing", "spatial-sharing-limited"]):
            throughput_values = data[exp].finetuning_throughputs[model_key]
            print(f"Finetuning throughput - {exp}")
            print(throughput_values)
            ax.plot(arrival_rates, throughput_values, marker=markers[idx], label=exp if j == 0 else None)
        ax.set_xticks(arrival_rates)
        ax.set_ylim(0)
        if j == 0:
            ax.set_ylabel("Finetuning Throughput\n(tokens/sec)")
        ax.grid(True)

        # Row 2: Inference Throughput
        ax = axs[2, j]
        for idx, exp in enumerate(["coserving", "spatial-sharing", "spatial-sharing-limited"]):
            throughput_values = data[exp].inference_throughputs[model_key]
            ax.plot(arrival_rates, throughput_values, marker=markers[idx], label=exp if j == 0 else None)
        ax.set_xticks(arrival_rates)
        ax.set_xlabel("Arrival Rate (QPS)")
        ax.set_ylim(0)
        if j == 0:
            ax.set_ylabel("Inference Throughput\n(tokens/sec)")
        ax.grid(True)

    # Adjust bottom margin to allocate space for the legend outside the subplots
    plt.subplots_adjust(bottom=0.2)
    # Add one common legend below the whole plot using handles from the first column
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["Collie", "Spatial Sharing (unlimited)", "Spatial Sharing (Token-/Layer-level finetuning)"],
            loc="lower center", fontsize=12, ncol=2)
    plt.savefig(os.path.join(output_folder, "spatial_sharing_comparison.pdf"), dpi=300)

##########################################################################
############ E2E comparison with Temporal Sharing PLOT    ################
##########################################################################
fig, axs = plt.subplots(3, 3, figsize=(12, 7))
model_names_simplified = ["Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"]
markers = ['o', 's', '^', 'D', 'v', 'p']  # Different dot types for the 6 curves

for j, model_key in enumerate(model_keys):
    model_name = model_names_simplified[j]
    # Only add the column title once at the top of each column
    axs[0, j].set_title(f"{model_name} (TP={tp_degrees[j]})", fontsize=12)
    
    # Row 0: SLO Attainment
    ax = axs[0, j]
    for idx, exp in enumerate(["coserving", "temporal-sharing-64", "temporal-sharing-128", "temporal-sharing-512", "spatial-sharing-limited"]):
        slo_values = data[exp].slo_attainments[model_key]
        ax.plot(arrival_rates, slo_values, marker=markers[idx], label=exp if j == 0 else None)
    ax.set_xticks(arrival_rates)
    ax.set_ylim(0, 1)
    if j == 0:
        ax.set_ylabel("\nSLO Attainment (%)", fontweight='bold')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # Row 1: Finetuning Throughput
    ax = axs[1, j]
    for idx, exp in enumerate(["coserving", "temporal-sharing-64", "temporal-sharing-128", "temporal-sharing-512", "spatial-sharing-limited"]):
        throughput_values = [4*x for x in data[exp].finetuning_throughputs[model_key]]
        ax.plot(arrival_rates, throughput_values, marker=markers[idx], label=exp if j == 0 else None)
        print(f"Finetuning throughput - {exp} - {model_key}: {throughput_values}")
    ax.set_xticks(arrival_rates)
    ax.set_ylim(0)
    if j == 0:
        ax.set_ylabel("Finetuning\nThroughput\n(tokens/sec)", fontweight='bold')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # Row 2: Inference Throughput
    ax = axs[2, j]
    for idx, exp in enumerate(["coserving", "temporal-sharing-64", "temporal-sharing-128", "temporal-sharing-512", "spatial-sharing-limited"]):
        throughput_values = [4*x for x in data[exp].inference_throughputs[model_key]]
        ax.plot(arrival_rates, throughput_values, marker=markers[idx], label=exp if j == 0 else None)
    ax.set_xticks(arrival_rates)
    ax.set_xlabel("Arrival Rate (req/s)", fontweight='bold')
    ax.set_ylim(0)
    if j == 0:
        ax.set_ylabel("Inference\nThroughput\n(tokens/sec)", fontweight='bold')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# Add one common legend below the whole plot using handles from the first column
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, ["Co-serving", "Temporal Sharing (freq=64)", "Temporal Sharing (freq=128)", "Temporal Sharing (freq=512)", "Spatial Sharing"],
           loc="upper center", fontsize=12, ncol=3)
plt.savefig(os.path.join(output_folder, "space_temporal_sharing_comparison.pdf"), dpi=300, bbox_inches='tight')