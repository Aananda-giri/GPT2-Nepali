[ ] Train on IRRIS nepali dataset
[ ] Train on multiple gpus
[ ] Train on TPUs

# -----------------------
# Nov. 23
# -----------------------

* 512 context length
* hugginface does not support renaming files accordint to: https://github.com/huggingface/transformers/issues/2561
* [X] so, renaming locally and then uploading to huggingface
* [X] then delete the old files from huggingface

i.e. pre_tokenized/nepberta.parquet -> pre_tokenized/nepberta_1024.parquet

* [X] and create new dataset with 512 context length: pre_tokenized/nepberta_512.parquet


# loading modified dataset
Examples of two file names:
https://huggingface.co/datasets/Aananda-giri/nepali_llm_datasets/resolve/main/pre_tokenized/nepberta.parquet
https://huggingface.co/datasets/Aananda-giri/nepali_llm_datasets/resolve/main/pre_tokenized/nepberta_test.parquet

we have to use [nepberta_512.parquet or nepberta_1024.parquet] in place of [nepberta.parquet] in loading script from previous_chapters/create_dataloader_v2


1. [X] save modified pre_tokenized.ipynb

2. [X] Test load_data from previous_chapter for context_length=512 and context_length=1024

3. [X] Run  kaggle v5-512: https://www.kaggle.com/code/mokinjay/sabastian-v5-512/edit


# -----------------------
# Nov. 12
# -----------------------
* calculate estimated training time

# -----------------------
# Nov. 12
# -----------------------

save tokenized data to huggingface
load from huggingface
initialize dataloader
modify bells_n_whistle training loop
save at the end of each epoch
use context_length of 512 instead of 1024?

```    
for epoch in epochs:
    for data in datas:
        ...
``` 

* schedule: load latest model from huggingface and resume training

Tokenizer
Dataset

# -----------------------
# Nov. 11
# -----------------------

[X] delete huggingface/nepberta 500Mb chunks
[X] upload new 50mb chunks
[X] set train test configs
re-write README examples and explaination

[X] code fix: append to .parquet
[X] tokenize and save to .parquet file
upload .parquet file to huggingface




# Dataloader

- upload entire dataset to huggingface (nepberta + wiki + crawled + ...)

  - required fields: url, text <cleaned_text>

- dataloader class:
  - pre-tokenize
  - __len__, __iterator__
  - Convert dataset into chunks
    - find ideal chunk size
    - can be tokenized at once with 16GB RAM: 45 million characters
    - sys.getsizeof(45 million characters) is equivalent to 80 Mb.
    - 50Mb chunks should be good enough

Dataset from huggingface and dataloader using pytorch?
num_workers=4?

# -----------------------
# Nov. 10
# -----------------------

# Dataloader

[ ] problem: cant load data more than 45M characters dataset because it attempts to tokenize all dataset at once
_ solution1: pre-tokenize
_ solution2: use map and avoid tokenizing all at once

[ ] training:
_ Save and load from huggingface and resume training ()
[ ] add epochs information as model_name

```    
for epoch in epochs:
    for data in datas:
        ...
```
