import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import defaultdict

def process_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Remove rows where is_warmup_step==1 or run_idx==0
    df = df[(df['is_warmup_step'] != 1) & (df['run_idx'] != 0)]
    start_time = df.iloc[0]['timestamp']
    df['timestamp'] = df['timestamp'] - start_time
    
    # Keep only the specified columns
    columns_to_keep = ['timestamp', 'num_prefilling_tokens', 'num_decoding_tokens', 'num_finetuning_fwd_tokens']
    df = df[columns_to_keep]
    
    # Create 5-second time bins (5,000,000 microseconds)
    # Floor division to group timestamps into 5-second intervals
    df['time_bin'] = (df['timestamp'] // 5000000) * 5000000


    
    # Group by time bins and aggregate
    grouped = df.groupby('time_bin').agg({
        'num_prefilling_tokens': 'sum',
        'num_decoding_tokens': 'sum', 
        'num_finetuning_fwd_tokens': 'sum',
    }).reset_index()
    
    # Create the inference and finetuning columns
    grouped['inference'] = (grouped['num_prefilling_tokens'] + grouped['num_decoding_tokens'])/5.0
    grouped['finetuning'] = (grouped['num_finetuning_fwd_tokens'])/5.0

    # Keep only the final columns we need
    final_df = grouped[['time_bin', 'inference', 'finetuning']]
    
    # Rename time_bin back to timestamp for clarity
    final_df = final_df.rename(columns={'time_bin': 'timestamp'})

    final_df['timestamp'] = final_df['timestamp'] / 1000000
    
    return final_df


def make_plot_a(json_file_path, output_folder):
    interval = 5  # 5 seconds
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract arrival times
    arrival_times = [entry['arrival_time'] for entry in data['entries']]
    
    if not arrival_times:
        print("No data found in the file")
        return
    
    # Find the time range
    min_time = min(arrival_times)
    max_time = max(arrival_times)
    
    # Create time bins (5-second intervals)
    time_bins = np.arange(min_time, max_time + interval, interval)
    
    # Count requests in each bin
    bin_counts = defaultdict(int)
    
    for arrival_time in arrival_times:
        # Find which bin this arrival time belongs to
        bin_index = int((arrival_time - min_time) // interval)
        bin_counts[bin_index] += 1
    
    # Prepare data for plotting
    time_points = [0]
    rates = [0]
    
    for i in range(len(time_bins) - 1):
        # Use the middle of each interval as the x-coordinate
        time_point = time_bins[i] + interval / 2
        # Calculate rate as requests per second (count / interval)
        rate = bin_counts[i] / interval
        
        time_points.append(time_point)
        rates.append(rate)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, rates, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Arrival Rate (req/s)')
    plt.yticks(range(5))
    plt.xticks(range(0, 660, 60))
    
    # plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "fig12a.pdf"), dpi=300, bbox_inches='tight')

def make_plot_b(df_processed, output_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(df_processed['timestamp'], df_processed['inference'], label='Inference', color='orange')
    plt.plot(df_processed['timestamp'], df_processed['finetuning'], label='Finetuning', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (tokens/sec)')
    plt.xticks(range(0, 660, 60))
    plt.yticks(range(0, 2300+575, 575))
    # plt.title('Inference and Finetuning Tokens over Time')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "fig12b.pdf"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # cd to directory of this file
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    fp_a = "./traces/burstgpt/qwen/sharegpt_8192_2.0_qps.json"
    if not os.path.exists(fp_a):
        raise FileNotFoundError("Fig 12a file does not exist!")
    
    make_plot_a(fp_a, output_folder)

    fp_b = './output/e2e/coserving/profiling/step_profiling_sharegpt_8192_2.0_qps_qwen_qwen2.5-14b-instruct_tensor_parallelism_2_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_70000_qps_0.000000_num_warmup_requests_10.csv'
    if not os.path.exists(fp_b):
        raise FileNotFoundError("Fig 12b file does not exist!")
    
    df_processed = process_csv(fp_b)
    make_plot_b(df_processed, output_folder)
