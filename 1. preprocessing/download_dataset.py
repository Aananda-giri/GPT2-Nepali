# download 1 chunk of size 500Mb from huggingface
from datasets import load_dataset

num_chunks_to_save = 1
target_dir = 'nepberta_sample'

# Load the dataset from the Hugging Face Hub
sampled_dataset_stream = load_dataset("Aananda-giri/nepberta-sample", split="train", streaming=True)

import os
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

# Save each chunk to a separate text file
for i in range(num_chunks_to_save):
    chunk = next(iter(sampled_dataset_stream))  # Get the next chunk
    with open(os.path.join(target_dir, f"combined_{i+1}.txt"), "w", encoding="utf-8") as file:
        file.write(chunk['text'])
    print(f"Saved chunk {i+1} to chunk_{i+1}.txt")