from datasets import load_dataset
from transformers import AutoTokenizer
import json, argparse, os
import matplotlib.pyplot as plt

def plot_histogram(data, title, cutoff_x, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, alpha=0.7)
    plt.axvline(x=cutoff_x, color='red', label='cutoff')
    plt.xlabel('Token Count')
    plt.ylabel('Number of Examples')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_and_print_stats(dataset, output_folder, max_length, dataset_title="T1 Original distribution"):
    # Plot distribution and calculate percentages for specific thresholds
    plot_histogram(dataset["token_count"], dataset_title, max_length, os.path.join(output_folder, f"{dataset_title.lower().replace(' ', '_')}.png"))
    thresholds = [1024, 2048, 4096, 8000, 8192, 16384]
    print("\nPercentage of examples below token thresholds:")
    for threshold in thresholds:
        count_below = sum(1 for count in dataset["token_count"] if count < threshold)
        percentage = (count_below / len(dataset["token_count"])) * 100
        print(f"< {threshold} tokens: {percentage:.2f}%")

def main(model_name, num_entries, max_length, seed, output_folder):
    # Load a sample dataset; adjust the dataset name and split as needed.
    dataset = load_dataset("NovaSky-AI/Sky-T1_data_17k", split="train")  # using a subset for speed
    dataset.shuffle(seed=seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def apply_template_and_count_tokens(example):
        # Apply the chat template to format the conversation
        messages = [{"role": "system", "content": example["system"]}] + [{"role": msg["from"], "content": msg["value"]} for msg in example["conversations"]]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # Tokenize the formatted text
        tokens = tokenizer.encode(tokenized_chat)
        
        # Save the number of tokens to a new field
        example["text"] = tokenized_chat
        example["token_count"] = len(tokens)
        return example
    tokenized_dataset = dataset.map(apply_template_and_count_tokens)
    avg_token_count = sum(tokenized_dataset["token_count"]) / len(tokenized_dataset)
    print(f"Average token count: {avg_token_count:.2f}")
    
    plot_and_print_stats(tokenized_dataset, output_folder, max_length, "T1 Original distribution")

    # Filter entries with token_count less than max_length.
    filtered_dataset = tokenized_dataset.filter(lambda example: example["token_count"] < max_length)
    filtered_dataset = filtered_dataset.select(range(num_entries))
    
    plot_and_print_stats(filtered_dataset, output_folder, max_length, "T1 Filtered distribution")

    # Extract the original text field from the filtered examples.
    text_list = filtered_dataset["text"]

    is_qwen = "qwen" in model_name.lower()
    is_llama = "llama" in model_name.lower()
    model_suffix = "_qwen" if is_qwen else "_llama" if is_llama else ""

    # Save the text list to a JSON file.
    flexllm_output_file=os.path.join(output_folder, f"t1{model_suffix}.json")
    with open(flexllm_output_file, "w") as f:
        json.dump(text_list, f, indent=2)
    
    # drop the "text column"
    filtered_dataset = filtered_dataset.remove_columns("text").remove_columns("token_count")
    llama_factory_output_folder = os.path.abspath("../LLaMA-Factory/data")
    llama_factory_output_file=os.path.join(llama_factory_output_folder, f"t1_flexllm{model_suffix}.json")
    with open(llama_factory_output_file, "w") as f:
        json.dump(list(filtered_dataset), f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build T1 training trace")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("-n", "--num_entries", type=int, default=1000, help="Number of entries")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed")
    parser.add_argument("-m", "--max-length", type=int, default=8192, help="Max dataset size")
    parser.add_argument("-o", "--output_folder", type=str, default="./traces", help="Output file name")
    args = parser.parse_args()

    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs("./traces", exist_ok=True)

    main(args.model_name, args.num_entries, args.max_length, args.seed, args.output_folder)
