import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from parse_data import BenchmarkResult

def load_benchmark_data(data_path=None):
    """Load benchmark data from pickle file."""
    if data_path is None:
        output_folder = "./output"
        data_path = os.path.join(output_folder, "benchmark_data.pkl")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Benchmark data file not found at {data_path}. Run parse_data.py first.")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"Benchmark data loaded from {data_path}")
    
    return data


def create_plots(data, output_folder=None):
    """Create all plots from the benchmark data."""
    
    if output_folder is None:
        output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Configuration
    models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
    tp_degrees = [1, 2, 4]
    qps_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    tpot_slos_ms = [45, 75, 75]
    ttft_slos_ms = [5000, 5000, 5000]
    
    experiment_types = [et for et in data.keys()]
    # Convert QPS values to floats for plotting
    arrival_rates = [4*float(q) for q in qps_values]
    # Get processed model names matching the keys in our data
    model_keys = [m.replace("/", "_").lower() for m in models]

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
    fig.legend(handles, ["Collie", "Separate cluster (25% vLLM - 75% LLaMA-Factory)", "Separate cluster (50% vLLM - 50% LLaMA-Factory)", "Separate cluster (75% vLLM - 25% LLaMA-Factory)"], loc="upper center", fontsize=12, ncol=2)
    plt.savefig(os.path.join(output_folder, "external_baselines.pdf"), dpi=300, bbox_inches='tight')

    ##########################################################################
    ######## E2E comparison with Temporal/Spatial Sharing PLOT    ############
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
        for idx, exp in enumerate(["coserving", "temporal-sharing-64", "temporal-sharing-128", "temporal-sharing-512", "spatial-sharing"]):
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
        for idx, exp in enumerate(["coserving", "temporal-sharing-64", "temporal-sharing-128", "temporal-sharing-512", "spatial-sharing"]):
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
        for idx, exp in enumerate(["coserving", "temporal-sharing-64", "temporal-sharing-128", "temporal-sharing-512", "spatial-sharing"]):
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
    plt.savefig(os.path.join(output_folder, "internal_baselines.pdf"), dpi=300, bbox_inches='tight')

    print(f"All plots saved to {output_folder}")



if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load the parsed data
    data = load_benchmark_data()
    
    # Create all plots
    create_plots(data)
