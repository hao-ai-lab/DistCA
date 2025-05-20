import random
import json
import os
import matplotlib.pyplot as plt
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def generate_fake():
    # 50000//8
    doclen_cnt = [
        96784, 1200, 350, 170, 110, 100, 80, 70, 
        60, 50 ,40 ,70, 60,30,80,70,40,20, 10,125
    ]
    doclen_avg = [
        50000//8 * (i + 1)
        for i in range(len(doclen_cnt))
    ]
    print(f"doclen_avg: {doclen_avg}")
    print(f"doclen_cnt: {doclen_cnt}")

    random.seed(42)
    dataset = []
    scope = 50000//8
    for i in range(len(doclen_cnt)):
        for j in range(doclen_cnt[i]):
            k = random.randint(i * scope + 1, (i+1) * scope)
            dataset.append(k)
    random.shuffle(dataset)

    return dataset

def plot(dataset, name):
    # Save the dataset as a JSON list
    with open(f'data/{name}.json', 'w') as f:
        json.dump(dataset, f)
    print(f"Saved dataset to data/{name}.json")

    import matplotlib.pyplot as plt
    plt.hist(dataset, bins=50, edgecolor='black')
    plt.yscale('log', base=2)
    plt.title('Dataset Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency (log2 scale)')
    plt.grid(True)

    # Save the histogram to a file
    plt.savefig(f'data/{name}.histogram.png')
    plt.close()



from multiprocessing import Pool, cpu_count
import numpy as np

def process_chunk(chunk):
    local_dataset = []
    for item in chunk:
        t = tokenizer(item["text"])['input_ids']
        t = len(t)
        if t != 0:
            local_dataset.append(t)
    return local_dataset

def generate_dataset(dataset_name) -> list[int]:
    # grab the dataset from huggingface
    dataset = []
    ds = load_dataset(dataset_name)
    
    # Define the number of chunks and the chunk size
    num_chunks = cpu_count()
    chunk_size = len(ds) // num_chunks
    
    # Split the dataset into chunks
    chunks = [ds[i:i + chunk_size] for i in range(0, len(ds), chunk_size)]
    
    with Pool(processes=num_chunks) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc=f"Generating dataset {dataset_name}"))
    
    # Flatten the list of lists into a single list
    dataset = [item for sublist in results for item in sublist]
    
    return dataset


if __name__ == "__main__":
    some_datasets = ["nvidia/OpenMathReasoning"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fake")
    args = parser.parse_args()
    print(f"Generating dataset {args.dataset}")

    if args.dataset == "fake":
        if os.path.exists("data/fake.json"):
            exit(0)
        dataset = generate_fake()
        plot(dataset, "fake")
    else:
        if os.path.exists(f"data/{args.dataset}.json"):
            exit(0)
        dataset = generate_dataset(args.dataset)
        plot(dataset, args.dataset)
