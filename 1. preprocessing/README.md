# step1: Load Dataset

- code: `0. load_data.ipynb`

# step2: Data cleaning

- although nepberta data was cleaned, there were still some non-devanagari charaters and some english text in the data.
- code: `2. pre-process.ipynb/CleanData`
- data cleaning steps:
  - Remove HTML tags
  - Normalize some characters
  - Convert 0-9 to ० - ९
  - Remove extra spaces

# Step-3: Pre tokenization

- combine the data from all pages with <|endoftext|> token in between, at the end.
- pre-tokenize with `stride=context_length=512`
- e.g. data before pre-tokenization:
  - `Text 1 \<endoftext\> Text 2 \<endoftext\> Text 3 \<endoftext\> ... Text N \<endoftext\>`

## **`Experiments/app.py`**<br>

- There were alot of characters. there are many varients of same characters that could be used to represent a character. e.g. two varienets of question mark: `？` and `?`

- this flask app helps visualize (with text example searched (`understanding_data.ipynb`) from original data) how different non-devanagari characters appear in text and replace \<input\> that saves characters and their replacement in `chars_to_replace.pkl`

## ToDo

- [ ] It would be nice to display frequency along with character in flask app : `app.py`
