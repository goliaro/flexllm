import os
import pandas as pd
import matplotlib.pyplot as plt

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
    grouped['inference'] = grouped['num_prefilling_tokens'] + grouped['num_decoding_tokens']
    grouped['finetuning'] = grouped['num_finetuning_fwd_tokens']
    
    # Keep only the final columns we need
    final_df = grouped[['time_bin', 'inference', 'finetuning']]
    
    # Rename time_bin back to timestamp for clarity
    final_df = final_df.rename(columns={'time_bin': 'timestamp'})

    final_df['timestamp'] = final_df['timestamp'] / 1000000
    
    return final_df

def make_plot1(df_processed, output_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(df_processed['timestamp'], df_processed['inference'], label='Inference', color='orange')
    plt.plot(df_processed['timestamp'], df_processed['finetuning'], label='Finetuning', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (tokens/sec)')
    # plt.title('Inference and Finetuning Tokens over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "fig12a.pdf"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # cd to directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    fp = './output/e2e/coserving/profiling/step_profiling_sharegpt_8192_2.0_qps_qwen_qwen2.5-14b-instruct_tensor_parallelism_2_max_requests_per_batch_256_max_tokens_per_batch_256_num_kv_cache_slots_70000_qps_0.000000_num_warmup_requests_10.csv'
    if not os.path.exists(fp):
        raise FileNotFoundError("Fig 12 file does not exist!")
    
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)
    
    df_processed = process_csv(fp)
    make_plot1(df_processed, output_folder)
