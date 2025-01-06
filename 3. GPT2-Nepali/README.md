
[ ] ToDo
[ ] dataloader not to load entire data to RAM
[ ] LLAMA2, GPT2 using huggingface transformer: train, inference

```
import pandas as pd
import os

def pre_process(text):
    # Example preprocessing function - modify according to your requirements
    text = text.lower()  # Convert to lowercase
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])  # Remove special characters
    text = ' '.join(text.split())  # Remove extra spaces
    return text

def save_to_txt_in_chunks(input_file, output_file, max_file_size_mb=5):
    # Size limit for the output file in bytes
    max_file_size = max_file_size_mb * 1024 * 1024

    # Read input CSV file in chunks
    chunk_size = 10000  # Number of rows per chunk
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)

    # Open the output file for writing
    with open(output_file, mode='w', encoding='utf-8') as f_out:
        for chunk in chunk_iter:
            # Process each chunk
            chunk['processed_text'] = chunk['text'].apply(pre_process)

            # Write each processed row to the output .txt file with <endoftext> token
            for row in chunk['processed_text']:
                # Append <endoftext> token
                f_out.write(f"{row} <endoftext>\n")

                # Check if the file size exceeds the limit
                if os.path.getsize(output_file) > max_file_size:
                    print(f"File size exceeded {max_file_size_mb} MB, stopping...")
                    return

if __name__ == "__main__":
    input_csv = 'dataset/clean_date_categories.csv'
    output_txt = 'output_data.txt'
    save_to_txt_in_chunks(input_csv, output_txt, max_file_size_mb=5)

```

### -------------------
* Using Approach-1
### -------------------

# Approach 1: Concatenate All Texts With \<endoftext\> Tokens
Description: 
* concatenate multiple texts, appending the \<endoftext\> token at the end of each one. The entire dataset becomes a long sequence of tokens, like this:
* Create training dataset on combined text

Example Dataset:

mathematica
Copy code
Text 1 \<endoftext\> Text 2 \<endoftext\> Text 3 \<endoftext\> ... Text N \<endoftext\>Approach 1: Concatenate All Texts With \<endoftext\> Tokens
Description: In this approach, you concatenate multiple texts, appending the \<endoftext\> token at the end of each one. The entire dataset becomes a long sequence of tokens, like this:

Used by: (llama, gpt) series

Example Dataset:
```
Text 1 <endoftext> Text 2 <endoftext> Text 3 <endoftext> ... Text N <endoftext>
```

## Pros:
* maximum utilization of sequence length without wasting padding tokens.
* Longer context learning: understand longer dependencies

## Cons:

* Loss of independence: model might treat separate texts as part of a continuous sequence. If the model overfits to cross-text dependencies, it might harm performance on independent tasks.
* Inconsistent context windows: Some texts might be shorter than the context window, while others exceed it, potentially cutting off parts of a text or extending the context unnecessarily over multiple texts.




# Approach 2: Separate Texts, Each Ending With \<endoftext\>
Description: 
* each text is processed independently, and the \<endoftext\> token is added at the end of each one.
* create training dataset separately for each text

```
Text 1 <endoftext>
Text 2 <endoftext>
Text 3 <endoftext>
...
Text N <endoftext>
```

* Used by: gemini

## Pros:

* Clear boundaries between texts
* No interference between texts: The model doesn't mistakenly form dependencies between different texts
* Easier batching: Each text's sequence can be handled separately in batches

## Cons:
* Inefficient use of sequence length: Shorter texts will waste sequence space (with padding) because each one is processed individually
* Loss of long-range dependencies: The model won't see as long a context as in Approach 1