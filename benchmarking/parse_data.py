from collections import defaultdict
import os
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import pickle
from tqdm import tqdm


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
    ttft = group.apply(lambda x: x[x["decoding_step_idx"] == 0]["timestamp"].values[0] - x[x["decoding_step_idx"] == -1]["timestamp"].values[0], include_groups=False)/1000
    # convert to milliseconds from microseconds
    return ttft.mean().iloc[1], ttft.median().iloc[1], ttft.quantile(0.99).iloc[1]


def get_queueing_time(df_original):
    df = df_original.copy()
    # remove entries where is_warmup_request is 1
    df = df[(df["is_warmup_request"] == 0)]
    group = df.groupby("request_guid", as_index=False)
    microsec_to_sec = 1_000_000
    # in each group, find the difference between the timestampt at request_step_idx=-1 and the timestamp at request_step_idx=-2.
    queueing_time = group.apply(lambda x: x[x["decoding_step_idx"] == -1]["timestamp"].values[0] - x[x["decoding_step_idx"] == -2]["timestamp"].values[0], include_groups=False)/1000
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


def check_file_availability(data, models, tp_degrees, kv_cache_slots_values, qps_values, llama_factory_model_names):
    """Check file availability for each experiment and report missing files."""
    print("Checking file availability for each experiment...")
    
    missing_files_report = {}
    all_missing_experiments = []
    
    for experiment_type, benchmark_result in data.items():
        total_files = 0
        missing_files = 0
        
        for i, model in enumerate(models):
            model_ = model.replace("/", "_").lower()
            tp_degree = tp_degrees[i]
            kv_cache_slots = kv_cache_slots_values[i]
            
            for qps in qps_values:
                if "vllm" not in experiment_type and "llama-factory" not in experiment_type:
                    # Check FlexLLM inference files
                    num_warmups = 10 if "coserving" in experiment_type else 0
                    filepath = os.path.join(benchmark_result.directory, 
                                          f"inference_request_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_{num_warmups}.csv")
                    total_files += 1
                    if not os.path.exists(filepath):
                        missing_files += 1
                    
                    # Check FlexLLM finetuning files
                    step_filepath = os.path.join(benchmark_result.directory, 
                                               f"step_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_{num_warmups}.csv")
                    total_files += 1
                    if not os.path.exists(step_filepath):
                        missing_files += 1
                        
                elif "vllm" in experiment_type:
                    # Check vLLM files
                    vllm_qps = 0
                    if "vllm-25pct" in experiment_type:
                        vllm_qps = round(qps*4, 1)
                    elif "vllm-50pct" in experiment_type:
                        vllm_qps = round(qps*2, 1)
                    elif "vllm-75pct" in experiment_type:
                        vllm_qps = round(qps*4/3, 1)
                        
                    filepath = os.path.join(benchmark_result.directory, 
                                          f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps.json")
                    if not os.path.exists(filepath):
                        filepath = os.path.join(benchmark_result.directory, 
                                              f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps_.json")
                    
                    total_files += 1
                    if not os.path.exists(filepath):
                        missing_files += 1
                        
                elif "llama-factory" in experiment_type:
                    # Check LLaMA-Factory files (only one file per model, not per QPS)
                    if qps == qps_values[0]:  # Only count once per model
                        llama_factory_model_name = llama_factory_model_names[i]
                        filepath = os.path.join(benchmark_result.directory, llama_factory_model_name, f"train_results.json")
                        total_files += 1
                        if not os.path.exists(filepath):
                            missing_files += 1
        
        missing_files_report[experiment_type] = (missing_files, total_files)
        print(f"- {experiment_type}: {missing_files}/{total_files} files missing")
        
        # Check if all files are missing for this experiment
        if total_files > 0 and missing_files == total_files:
            all_missing_experiments.append(experiment_type)
    
    # Exit with error if any experiment has all files missing
    if all_missing_experiments:
        print(f"\nERROR: The following experiments have ALL files missing:")
        for exp in all_missing_experiments:
            print(f"  - {exp}")
        print("Please check the data paths and ensure the benchmark results exist.")
        exit(1)
    
    print("File availability check completed.\n")
    return missing_files_report


def parse_benchmark_data():
    """Parse raw benchmark data from CSV and JSON files and save to pickle file."""
    
    # Configuration
    models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
    llama_factory_model_names = ["t1_llama_8B/lora/sft", "t1_qwen_14B/lora/sft", "t1_qwen_32B/lora/sft"]
    tp_degrees = [1, 2, 4]
    kv_cache_slots_values = [70000, 70000, 60000]
    qps_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    tpot_slos_ms = [45, 75, 75]
    ttft_slos_ms = [5000, 5000, 5000]

    output_folder = "./output"
    save_path = os.path.join(output_folder, "benchmark_data.pkl")

    if os.path.exists(save_path):
        print(f"Benchmark data already exists at {save_path}. Remove it to regenerate.")
        return save_path

    # Initialize data structure
    data = {
        "coserving": BenchmarkResult(directory="./output/e2e/coserving/profiling"),
        "spatial-sharing": BenchmarkResult(directory="./output/e2e/spatial_sharing/profiling"),
        "temporal-sharing-64": BenchmarkResult(directory="./output/e2e/temporal_sharing/64/profiling"),
        "temporal-sharing-128": BenchmarkResult(directory="./output/e2e/temporal_sharing/128/profiling"),
        "temporal-sharing-512": BenchmarkResult(directory="./output/e2e/temporal_sharing/512/profiling"),
        "vllm-25pct": BenchmarkResult(directory="./output/vllm"),
        "vllm-50pct": BenchmarkResult(directory="./output/vllm"),
        "vllm-75pct": BenchmarkResult(directory="./output/vllm"),
        "llama-factory-25pct": BenchmarkResult(directory="./output/llama-factory"),
        "llama-factory-50pct": BenchmarkResult(directory="./output/llama-factory"),
        "llama-factory-75pct": BenchmarkResult(directory="./output/llama-factory"),
    }

    # Check file availability before processing
    check_file_availability(data, models, tp_degrees, kv_cache_slots_values, qps_values, llama_factory_model_names)

    # Parse data for each experiment type
    for experiment_type in data.keys():
        # print(f"Processing {experiment_type}...")
        
        total_iterations = len(models) * len(qps_values)
        progress_bar = tqdm(total=total_iterations, desc=f"Processing {experiment_type}", unit="iter")

        for i, model in enumerate(models):
            model_ = model.replace("/", "_").lower()
            tp_degree = tp_degrees[i]
            kv_cache_slots = kv_cache_slots_values[i]
            tpot_slo_ms = tpot_slos_ms[i]
            ttft_slo_ms = ttft_slos_ms[i]
            
            # Initialize lists for this model
            data[experiment_type].tpot_slo_attainments[model_] = []
            data[experiment_type].inference_throughputs[model_] = []
            data[experiment_type].finetuning_throughputs[model_] = []
            data[experiment_type].queueing_times[model_] = []
            data[experiment_type].ttfts[model_] = []
            data[experiment_type].slo_attainments[model_] = []
            
            for qps in qps_values:
                # Existing processing code for each qps value remains here
                if "vllm" not in experiment_type and "llama-factory" not in experiment_type:
                    num_warmups = 10 if "coserving" in experiment_type else 0
                    # Process FlexLLM data
                    filepath = os.path.join(data[experiment_type].directory, 
                                          f"inference_request_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_{num_warmups}.csv")

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
                    
                    # Process finetuning data
                    step_filepath = os.path.join(data[experiment_type].directory, 
                                               f"step_profiling_sharegpt_8192_{qps}_qps_{model_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_{kv_cache_slots}_qps_0.000000_num_warmup_requests_{num_warmups}.csv")

                    if os.path.exists(step_filepath):
                        df = pd.read_csv(step_filepath)
                        ft_throughput = get_ft_throughput(df)
                        data[experiment_type].finetuning_throughputs[model_].append(ft_throughput)
                    else:
                        print(f"File {step_filepath} does not exist.")
                        data[experiment_type].finetuning_throughputs[model_].append(np.nan)
                        
                elif "vllm" in experiment_type:
                    # Process vLLM data
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
                        
                    filepath = os.path.join(data[experiment_type].directory, 
                                          f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps.json")
                    if not os.path.exists(filepath):
                        filepath = os.path.join(data[experiment_type].directory, 
                                              f"results_sharegpt_eager_v1_{model_}_bz_256_max_num_batched_tokens_256_{vllm_qps:.1f}_qps_.json")
                    
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
                    # Process LLaMA-Factory data
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
                
                progress_bar.update(1)
        progress_bar.close()

    # Save parsed data
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Benchmark data saved to {save_path}")
    
    return save_path


if __name__ == "__main__":
    print("Parsing benchmark data...")
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = parse_benchmark_data()
    print(f"Data parsing complete. Saved to: {save_path}")
