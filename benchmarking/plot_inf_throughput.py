import os, json
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

qps=3.0
model_name="Qwen/Qwen2.5-14B-Instruct"
model_name_=model_name.replace("/", "_").lower()
tp_degree=2
kv_cache_slots=70000
output_directory="/global/homes/g/goliaro/flexllm/benchmarking/plots/throughput_vs_time"

def plot_flexllm_throughput():
    base_directory="/global/homes/g/goliaro/flexllm/benchmarking/output/e2e/coserving/profiling"

    step_filepath=os.path.join(base_directory, f"step_profiling_sharegpt_8192_{qps}_qps_{model_name_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_0.csv")
    assert os.path.isfile(step_filepath)
    

    df = pd.read_csv(step_filepath)
    df = df[df["is_warmup_step"] != 1]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_diff_sec"] = df["timestamp"].diff() / 1_000_000
    # Remove the first row with NaN time difference
    df = df.dropna()
    # Compute throughput: tokens per second = num_decoding_tokens / time_diff_sec
    df["throughput"] = df["num_decoding_tokens"] / df["time_diff_sec"]
    # Create a relative time axis in seconds (starting from 0)
    df["relative_time"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1_000_000
    plt.plot(df["relative_time"], df["throughput"])
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Throughput vs Time")
    # plt.legend()
    # plt.show()
    plt.savefig(os.path.join(output_directory, f"throughput_vs_time_{model_name_}_qps_{qps}.png"))

def plot_vllm_throughput():
    traces_base_directory="/global/homes/g/goliaro/flexllm/benchmarking/traces/burstgpt/qwen"
    trace_filepath=os.path.join(traces_base_directory, f"sharegpt_8192_{qps*2:.1f}_qps.json")
    with open(trace_filepath, 'r') as f:
        reqs = json.load(f)
    arrival_times = [entry['arrival_time'] for entry in reqs['entries']]

    # --- load the second JSON and extract ttfts and itls ---
    results_base_directory="/global/homes/g/goliaro/flexllm/benchmarking/output/vllm"
    results_filepath=os.path.join(results_base_directory, f"results_sharegpt_eager_v1_{model_name_}_bz_256_max_num_batched_tokens_256_{qps*2:.1f}_qps_.json")
    with open(results_filepath, 'r') as f:
        lat = json.load(f)
    ttfts = lat['ttfts']
    itls  = lat['itls']

    # --- sanity check lengths ---
    assert len(arrival_times) == len(ttfts) == len(itls), \
        "mismatch in number of requests vs. ttfts vs. itls"

    # --- assemble into a single DataFrame ---
    df = pd.DataFrame({
        'arrival_time': arrival_times,
        'ttft':          ttfts,
        'inter_token_latencies': itls
    })
    token_times = []
    for _, row in df.iterrows():
        start = row['arrival_time'] + row['ttft']
        # cumulative time for each output token
        cum_times = np.cumsum(row['inter_token_latencies'])
        token_times.extend(start + cum_times)

    token_times = np.array(token_times)

    # 2) Build a time‐binned histogram of tokens
    bin_width = 0.1  # seconds, adjust for smoother/coarser curves
    t_start = token_times.min()
    t_end   = token_times.max()
    bins = np.arange(t_start, t_end + bin_width, bin_width)

    counts, edges = np.histogram(token_times, bins=bins)

    # 3) Convert counts → throughput (tokens/sec)
    throughput = counts / bin_width
    bin_centers = edges[:-1] + bin_width / 2

    # 4) Plot
    plt.figure(figsize=(10, 4))
    plt.plot(bin_centers, throughput, drawstyle='steps-mid')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('LLM Output‐Token Throughput Over Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.show()
    plt.savefig(os.path.join(output_directory, f"vllm_throughput_vs_time_{model_name_}_qps_{qps}.png"))

plot_flexllm_throughput()
plot_vllm_throughput()
