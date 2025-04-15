import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import urllib.request
from dataclasses import asdict, dataclass, field
import json, os, random, requests, argparse
from tqdm.asyncio import tqdm
from typing import List, Optional
from collections import OrderedDict
from transformers import AutoTokenizer
import pandas as pd
from math import ceil
from random import uniform
import numpy as np

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

@dataclass
class TraceEntry:
    prompt: str
    response: str
    prompt_length: int
    response_length: int
    arrival_time: int

@dataclass
class TraceMetadata:
    num_warmup_requests: int
    avg_entries_per_partition: float
    max_prompt_length: int
    min_prompt_length: int
    avg_prompt_length: float
    max_response_length: int
    min_response_length: int
    avg_response_length: float
    avg_total_length: int
    max_total_length: int
    total_tokens: int
    trace_type: str
    arrival_rate: float

@dataclass
class Trace:
    entries: List[TraceEntry] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=lambda: TraceMetadata(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "offline", 0.0))

def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename

def get_slice(df, slice_duration, seed):
    # Assume df is your DataFrame with a "Timestamp" column
    min_time = df['Timestamp'].min()
    max_time = df['Timestamp'].max()
    # Ensure there is enough room for the slice
    max_start_time = max_time - slice_duration
    if max_start_time <= min_time:
        raise ValueError(f"The dataset is not long enough for a {slice_duration/60}-minute slice.")
    # Randomly pick a start time between min_time and max_start_time
    np.random.seed(seed)
    random_start_time = np.random.uniform(min_time, max_start_time)
    slice_end_time = random_start_time + slice_duration
    # Extract the slice of data
    df_slice = df[(df['Timestamp'] >= random_start_time) & (df['Timestamp'] < slice_end_time)].copy()
    # subtract the start time from the timestamps to get relative time
    df_slice['Timestamp'] = df_slice['Timestamp'] - random_start_time

    return df_slice

def scale_arrival_rate_fixed_duration(df_slice, target_rate_sec):
    """
    Scale the arrival rate by sampling requests to match a target rate while
    preserving the distribution of interarrival times and maintaining the same duration.
    
    Args:
        df_slice: DataFrame with a 'Timestamp' column
        target_rate: Desired arrival rate in requests/second
        
    Returns:
        DataFrame with sampled requests to match the target rate
    """
    # Calculate the current arrival rate
    num_requests = len(df_slice)
    duration = df_slice['Timestamp'].max()  # assumes timestamps start at 0
    current_rate = num_requests / duration
    
    # Calculate how many requests we need for the target rate
    target_requests = int(target_rate_sec * duration)
    
    # If we need fewer requests (downsampling)
    if target_requests < num_requests:
        # Sample the required number of requests while preserving timestamp order
        df_scaled = df_slice.sample(n=target_requests, random_state=42).sort_values('Timestamp').reset_index(drop=True)
    
    # If we need more requests (upsampling) - This is more complicated
    elif target_requests > num_requests:
        # We need to duplicate some requests
        repeats = int(np.ceil(target_requests / num_requests))
        dfs = [df_slice]
        
        for i in range(1, repeats):
            # Create a copy with very small time variations to avoid exact duplicates
            df_copy = df_slice.copy()
            # Add tiny random variations (less than minimum interarrival time)
            min_interval = df_slice['Timestamp'].diff().min()
            jitter = np.random.uniform(0, min_interval/10, size=len(df_copy))
            df_copy['Timestamp'] = df_copy['Timestamp'] + jitter
            dfs.append(df_copy)
            
        # Concatenate all copies
        df_concat = pd.concat(dfs, ignore_index=True)
        # Sample the exact number of requests needed and sort by timestamp
        df_scaled = df_concat.sample(n=target_requests, random_state=42).sort_values('Timestamp').reset_index(drop=True)
    
    else:  # target_requests == num_requests
        df_scaled = df_slice.copy()
    
    print(f"Original trace: {num_requests} requests over {duration:.2f}s at {current_rate:.3f} req/s")
    print(f"Scaled trace: {len(df_scaled)} requests over {duration:.2f}s at {target_rate_sec:.3f} req/s")
    
    # subtract the start time from the timestamps to get relative time
    df_scaled['Timestamp'] = df_scaled['Timestamp'] - df_scaled['Timestamp'].min()

    return df_scaled

def plot_histogram(df_hist, filepath, bin_width=5, tokens=False):
    """
    Plot a histogram of request arrival rates.
    
    Args:
        df_hist: DataFrame with a 'Timestamp' column
        bin_width: Width of histogram bins in seconds
        filename: Output filename for the plot
        title: Custom title for the plot (if None, a default will be used)
        max_time: Maximum x-axis limit (if None, will use max timestamp in data)
    """
    # Calculate the actual duration from the data
    max_time = df_hist['Timestamp'].max()
    # Define the bin width and create bin edges
    bins = np.arange(0, max_time + bin_width, bin_width)
    # Use numpy's histogram function to count arrivals in each bin
    counts, _ = np.histogram(df_hist['Timestamp'], bins=bins)

    # Convert counts to arrival rate in req/s by dividing by the bin width
    arrival_rates = counts / bin_width
    token_sums, _ = np.histogram(df_hist['Timestamp'], bins=bins, weights=df_hist['response_len'])
    # Compute the token arrival rate in tokens/s by dividing by the bin width (5s)
    token_rate = token_sums / bin_width


    if tokens:
        y_axis = token_rate
        y_label = 'Requested throughput (tokens/s)'
    else:
        y_axis = arrival_rates
        y_label = 'Requested rate (req/s)'

    # Compute bin centers
    bin_centers = bins[:-1] + bin_width / 2

    duration_minutes = max_time / 60
    measured_req_rate = len(df_hist) / df_hist['Timestamp'].max()
    title = f'Requested throughput over {duration_minutes:.1f} minutes (averaged over {bin_width}-s intervals)\nRequest rate: {measured_req_rate:.2f} req/s'
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, y_axis, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim(0, max_time)
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()  # Close the figure to free memory

def get_burstgpt_timestamps(slice_duration_sec, output_folder, seed):
    dataset_path = os.path.join(output_folder, "..", "BurstGPT_without_fails_2.csv")
    # if a file does not exist at the dataset path, download it from https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
    if not os.path.exists(dataset_path):
        download_url = "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv"
        print(f"Downloading dataset from {download_url} to {dataset_path}...")
        urllib.request.urlretrieve(download_url, dataset_path)
    df = pd.read_csv(dataset_path)
    df = df[df["Model"] == "ChatGPT"]
    # drop the columns "Model", "Request tokens", "Total tokens", "Log Type"
    df = df.drop(columns=["Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"])
    return get_slice(df, slice_duration_sec, seed)

def add_sharegpt_prompts_and_responses(timestamps_df, tokenizer, max_length, seed, apply_chat_template=False):
    # Add prompt, response, prompt_len, response_len columns to the dataframe
    timestamps_df['prompt'] = "empty"
    timestamps_df['response'] = "empty"
    timestamps_df['prompt_len'] = 0
    timestamps_df['response_len'] = 0

    
    # Load the ShareGPT dataset.
    dataset_path = download_and_cache_file(SHAREGPT_URL)
    with open(dataset_path) as f:
        dataset = json.load(f, object_pairs_hook=OrderedDict)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
        if data["conversations"][0]["from"] == "human" and data["conversations"][1]["from"] == "gpt"
    ]
    # Shuffle the dataset.
    random.seed(seed)
    random.shuffle(dataset)

    # First, reset the DataFrame index to have sequential integers
    timestamps_df = timestamps_df.reset_index(drop=True)

    # Iterate over the rows in order
    for idx in tqdm(range(len(timestamps_df)), desc="Processing rows"):
        if idx >= len(dataset):
            break

        prompt = dataset[idx][0]
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        response = dataset[idx][1]
        prompt_length = len(tokenizer(prompt)["input_ids"])
        response_length = len(tokenizer(response)["input_ids"])
        
        # If the combined token length exceeds the maximum, skip updating this row
        if prompt_length + response_length > max_length:
            continue
        
        # Set the values for the corresponding row in the DataFrame
        timestamps_df.at[idx, 'prompt'] = prompt
        timestamps_df.at[idx, 'response'] = response
        timestamps_df.at[idx, 'prompt_len'] = prompt_length
        timestamps_df.at[idx, 'response_len'] = response_length

    # Optionally, assert that all rows have been processed as expected
    assert len(timestamps_df) <= len(dataset)
    return timestamps_df

def save_trace(df, qps, filepath):
    # populate Trace datapath from df
    entries = []
    for _, row in df.iterrows():
        entry = TraceEntry(
            prompt=row['prompt'],
            response=row['response'],
            prompt_length=row['prompt_len'],
            response_length=row['response_len'],
            arrival_time=round(row['Timestamp'], 3)
        )
        entries.append(entry)
    # Create Trace object
    trace = Trace(entries=entries)
    # add metadata
    # trace.metadata.num_warmup_requests = 0
    # trace.metadata.avg_entries_per_partition = len(trace.entries)
    # trace.metadata.max_prompt_length = df['prompt_len'].max()
    # trace.metadata.min_prompt_length = df['prompt_len'].min()
    # trace.metadata.avg_prompt_length = df['prompt_len'].mean()
    # trace.metadata.max_response_length = df['response_len'].max()['response_len']
    # trace.metadata.min_response_length = df['response_len'].min()['response_len']
    # trace.metadata.avg_response_length = df['response_len'].mean()['response_len']
    # trace.metadata.avg_total_length = (df['prompt_len'] + df['response_len']).mean()
    # trace.metadata.max_total_length = (df['prompt_len'] + df['response_len']).max()
    # trace.metadata.total_tokens = df['prompt_len'].sum() + df['response_len'].sum()
    # trace.metadata.trace_type = "burstgpt"
    # trace.metadata.arrival_rate = qps

    trace_dict = asdict(trace)
    with open(filepath, 'w') as f:
        json.dump(trace_dict, f, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BurstGPT (ShareGPT) trace")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("-m", "--max-length", type=int, default=8192, help="Maximum prompt + response length")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-d", "--duration", type=int, default=10, help="Slice duration in minutes")
    parser.add_argument("-o", "--output_folder", type=str, default="./traces/burstgpt", help="Output folder path")
    parser.add_argument("-q", "--qps", type=float, default=5.0, help="Arrival Rate in req/s")
    args = parser.parse_args()

    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    model_subfolder = "unknown"
    if "qwen" in args.model_name.lower():
        model_subfolder = "qwen"
    elif "llama" in args.model_name.lower():
        model_subfolder = "llama"

    output_folder = os.path.join(args.output_folder, model_subfolder)
    os.makedirs(output_folder, exist_ok=True)

    timestamps_df = get_burstgpt_timestamps(slice_duration_sec=args.duration*60, output_folder=output_folder, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df_slice = add_sharegpt_prompts_and_responses(timestamps_df, tokenizer, args.max_length, seed=args.seed, apply_chat_template=False)
    
    # Plot original slice
    plot_histogram(df_slice, filepath=os.path.join(output_folder, f"sharegpt_{args.max_length}_original_req.png"))
    plot_histogram(df_slice, filepath=os.path.join(output_folder, f"sharegpt_{args.max_length}_original_thr.png"), tokens=True)

    df_scaled = scale_arrival_rate_fixed_duration(df_slice, target_rate_sec=args.qps)
    df_scaled2 = add_sharegpt_prompts_and_responses(df_scaled, tokenizer, args.max_length, seed=args.seed, apply_chat_template=False)
    plot_histogram(df_scaled2, filepath=os.path.join(output_folder, f"sharegpt_{args.max_length}_{args.qps:.2}_qps_req.png"))
    plot_histogram(df_scaled2, filepath=os.path.join(output_folder, f"sharegpt_{args.max_length}_{args.qps:.2}_qps_thr.png"), tokens=True)
    
    save_trace(df_scaled2, qps=args.qps, filepath=os.path.join(output_folder, f"sharegpt_{args.max_length}_{args.qps:.2}_qps.json"))
