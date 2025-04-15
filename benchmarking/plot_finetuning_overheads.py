import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

expected_fwd_tokens = [0,128,256,384,512]
expected_bwd_layers = [0, 8, 16, 24, 32]

def get_run_idx(max_tokens_per_batch, num_fwd_finetuning_tokens, num_bwd_layers):
    
    fwd_idx = expected_fwd_tokens.index(num_fwd_finetuning_tokens)
    bwd_idx = expected_bwd_layers.index(num_bwd_layers)
    return 1 + fwd_idx * len(expected_bwd_layers) + bwd_idx

def get_req_guids(max_tokens_per_batch, num_fwd_finetuning_tokens, num_bwd_layers):
    num_warmup_requests=11
    max_requets_per_batch=8

    fwd_idx = expected_fwd_tokens[max_tokens_per_batch].index(num_fwd_finetuning_tokens)
    bwd_idx = expected_bwd_layers[max_tokens_per_batch].index(num_bwd_layers)
    
    # Add warmup requests
    start_guid = num_warmup_requests
    # Add requests from runs with 0 fwd tokens, where each group has 8 requests
    if fwd_idx == 0:
        start_guid += max_requets_per_batch * bwd_idx
    else:
        start_guid += max_requets_per_batch * len(expected_bwd_layers[max_tokens_per_batch])
        # Add requests from runs with >= 1 fwd tokens
        start_guid += ((fwd_idx-1) * len(expected_bwd_layers[max_tokens_per_batch]) + bwd_idx) * (max_requets_per_batch+1)
    guids=[1000000+x for x in range(start_guid, start_guid+max_requets_per_batch)]
    return guids



def plot_fwd_overhead(filepath, model_name, tp_degree, bz, num_tokens_per_batch):
    # Load the CSV file
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    # Plot one line for each bwd token value
    for fwd_tokens in expected_fwd_tokens:
        for bwd_layers in expected_bwd_layers:
            run_idx = get_run_idx(num_tokens_per_batch, fwd_tokens, bwd_layers)
            print(f"Run {run_idx}: {fwd_tokens} fwd tokens, {bwd_layers} bwd layers")

    df = df[(df['run_idx'] >= 1) & (df['is_warmup_step'] != 1)]
    # First, sort the dataframe and compute the step_time for each run_idx group
    df = df.sort_values(['run_idx', 'timestamp'])
    df['step_time'] = df.groupby('run_idx')['timestamp'].diff() / 1000

    # For each run_idx group, compute the mean step_time for the rows matching the criteria per fwd_tokens
    mean_steps = {fwd: [] for fwd in expected_fwd_tokens}
    for run_idx_val, group in df.groupby('run_idx'):
        for fwd_tokens in expected_fwd_tokens:
            sel = group[
                (group['num_decoding_tokens'] == 256) &
                (group['num_prefilling_tokens'] == 0) &
                (group['num_finetuning_fwd_tokens'] == fwd_tokens) &
                (group['num_finetuning_bwd_tokens'] == 0)
            ]
            mean_val = sel['step_time'].mean()
            if not np.isnan(mean_val):
                mean_steps[fwd_tokens].append(mean_val)

    # Now, average the means across all run_idx groups for each fwd_tokens value
    latencies = [np.mean(mean_steps[fwd]) if mean_steps[fwd] else np.nan for fwd in expected_fwd_tokens]
    plt.plot(expected_fwd_tokens, latencies, marker='o',)

    plt.xticks(expected_fwd_tokens) 
    plt.ylim(0)
    plt.title(f"Batch Step Time vs Number of Added Forward Finetuning Tokens\nModel: {model_name} (TP={tp_degree})\nInference Batch Size: {bz} - Inference Max Tokens per Batch: {num_tokens_per_batch}\n")
    plt.xlabel('Number of forward finetuning tokens')
    plt.ylabel('Average step time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    model_name_ = model_name.replace("/", "_").lower()
    plt.savefig(f'./plots/overhead_test/fwd_overhead_{model_name_}.pdf', bbox_inches='tight')
    

def plot_bwd_overhead(filepath, model_name, tp_degree, bz, num_tokens_per_batch, ft_bwd_tokens):
    # Load the CSV file
    df = pd.read_csv(filepath)

    plt.figure(figsize=(10, 6))
    # # Plot one line for each bwd token value
    # for fwd_tokens in expected_fwd_tokens[num_tokens_per_batch]:
    #     for bwd_layers in expected_bwd_layers[num_tokens_per_batch]:
    #         run_idx = get_run_idx(num_tokens_per_batch, fwd_tokens, bwd_layers)
    #         print(f"Run {run_idx}: {fwd_tokens} fwd tokens, {bwd_layers} bwd layers")

    df = df[(df['run_idx'] >= 1) & (df['is_warmup_step'] != 1)]
    # First, sort the dataframe and compute the step_time for each run_idx group
    df = df.sort_values(['run_idx', 'timestamp'])
    df['step_time'] = df.groupby('run_idx')['timestamp'].diff() / 1000
    mean_steps = {bwd: [] for bwd in expected_bwd_layers}
    for run_idx_val, group in df.groupby('run_idx'):
        for bwd_layers in expected_bwd_layers:
            sel = group[
                (group['num_decoding_tokens'] == 256) &
                (group['num_prefilling_tokens'] == 0) &
                (group['num_finetuning_fwd_tokens'] == 0) &
                (group['num_finetuning_bwd_tokens'] == 4096) &
                (group['num_bwd_layers']== bwd_layers)
            ]
            mean_val = sel['step_time'].mean()
            if not np.isnan(mean_val):
                mean_steps[bwd_layers].append(mean_val)

    latencies = [np.mean(mean_steps[bwd]) if mean_steps[bwd] else np.nan for bwd in expected_bwd_layers]
    plt.plot(expected_bwd_layers, latencies, marker='o',)

    plt.xticks(expected_bwd_layers) 
    plt.title(f"Batch Step Time vs Number of Backward Finetuning Layers\nModel: {model_name} (TP={tp_degree})\nInference Batch Size: {bz} - Inference Max Tokens per Batch: {num_tokens_per_batch}\nBWD finetuning tokens: {ft_bwd_tokens}")
    plt.xlabel('Number of backward finetuning layers')
    plt.ylim(0)
    plt.ylabel('Average step time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    model_name_ = model_name.replace("/", "_").lower()
    plt.savefig(f'./plots/overhead_test/bwd_overhead_{model_name_}.pdf', bbox_inches='tight')
    

if __name__ == "__main__":

    # Change working directory to folder containing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Make plots directory if it doesn't exist
    if not os.path.exists('./plots/overhead_test'):
        os.makedirs('./plots/overhead_test')
    model_names=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct"
    ]
    model_names_=[model_name.replace("/", "_").lower() for model_name in model_names]
    tp_degrees = [1,2,4]

    ft_bwd_tokens=4096
    bz=256
    tokens_per_batch=1024

    for i, model_name in enumerate(model_names):
        tp_degree = tp_degrees[i]
        model_name_ = model_names_[i]
        fp_step=f"/global/homes/g/goliaro/flexllm/benchmarking/output/overhead_test/profiling/step_profiling_overhead_test_{model_name_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_{bz}_max_tokens_per_batch_{tokens_per_batch}_num_kv_cache_slots_60000_qps_0.000000_num_warmup_requests_10.csv"
        fp_req=f"/global/homes/g/goliaro/flexllm/benchmarking/output/overhead_test/profiling/inference_request_profiling_overhead_test_{model_name_}_tensor_parallelism_{tp_degree}_max_requests_per_batch_{bz}_max_tokens_per_batch_{tokens_per_batch}_num_kv_cache_slots_60000_qps_0.000000_num_warmup_requests_10.csv"
        plot_fwd_overhead(fp_step, model_name, tp_degree, bz, tokens_per_batch)
        plot_bwd_overhead(fp_step, model_name, tp_degree, bz, tokens_per_batch, ft_bwd_tokens)