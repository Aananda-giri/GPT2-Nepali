# Tokenizer

Nepali CharBPETokenizer with 50k vocab_size = 50k trained on nepberta dataset.

- huggingface tokenizers library was used to train a CharBPETokenizer on cleaned dataset from nepberta.

- tried tokenizes with different vocab sizes (30k, 50k, 75k) and selected 50k for providing balance between small vocab size and quality of tokens.

## Usage:

```python

# loading from huggingface
# -------------------------

from transformers import PreTrainedTokenizerFast

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/NepaliBPE")

# Example usage
text = "तरुल लगाएर खासै नाफा त छैन तर धेरै मिहिनेत गर्नु पनि पर्दैन।"

# Tokenize
print(f"Tokenized: {tokenizer.tokenize(text)}")
# Tokenized: ['तरुल</w>', 'लगाएर</w>', 'खासै</w>', 'नाफा</w>', 'त</w>', 'छैन</w>', 'तर</w>', 'धेरै</w>', 'मिहिनेत</w>', 'गर्नु</w>', 'पनि</w>', 'पर्दैन</w>', '।</w>']

# Encode
tokens = tokenizer.encode(text)
print("Tokens:", tokens)
# Tokens: [29513, 3554, 4918, 4332, 155, 665, 495, 853, 13242, 1569, 338, 5917, 276]

# Decode
print("Decoded:", tokenizer.decode(tokens))
# Decoded: तरुल लगाएर खासै नाफा त छैन तर धेरै मिहिनेत गर्नु पनि पर्दैन ।

```

```python

# load from local directory
# -------------------------
from transformers import PreTrainedTokenizerFast

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("NepaliBPE")

# Example usage
text = "तरुल लगाएर खासै नाफा त छैन तर धेरै मिहिनेत गर्नु पनि पर्दैन।"

# Tokenize
print(f"Tokenized: {tokenizer.tokenize(text)}")
# Tokenized: ['तरुल</w>', 'लगाएर</w>', 'खासै</w>', 'नाफा</w>', 'त</w>', 'छैन</w>', 'तर</w>', 'धेरै</w>', 'मिहिनेत</w>', 'गर्नु</w>', 'पनि</w>', 'पर्दैन</w>', '।</w>']

# Encode
tokens = tokenizer.encode(text)
print("Tokens:", tokens)
# Tokens: [29513, 3554, 4918, 4332, 155, 665, 495, 853, 13242, 1569, 338, 5917, 276]

# Decode
print("Decoded:", tokenizer.decode(tokens))
# Decoded: तरुल लगाएर खासै नाफा त छैन तर धेरै मिहिनेत गर्नु पनि पर्दैन ।
```

## `tokenizer.py`

This file contains main code for training tokenizer.

## `tokenizer.ipynb`:

This file contains code for preparing dataset, training tokenizer and pushing it to huggingface hub.

## vocab_size choice

- GPT2 uses a vocabulary size of 50,257, which consists of 50,000 Byte Pair Encoding (BPE) merges, 256 possible byte values for encoding raw characters, and one special token (<|endoftext|>). The choice of 50,257 does not seem to be related with training efficiency.

- For a Nepali language tokenizer, a vocabulary size of 50,000 seems to be good enough.

## `token-visualization/`

- Basic clone of [tiktokenizer](https://tiktokenizer.vercel.app/) to visualize tokens using tokenizer we previously trained.
- built using flask.

- `tokenizer.py` (modified `archive/tokenizer.py`) is the main file for the tokenizer

- please refer to `archive/` for hit and trials (including token visualization)

## `nep_minbpe/`

- modified code from [karpathy/minbpe](https://github.com/karpathy/minbpe)for nepali characters
- yet to do: special tokens, regex patterns
- built using flask.

# Nepali BPE tokenizer:

- `tokenize.ipynb` is final version of tokenizer we ended up with which uses code from `tokenize.py`.

- **Training code**: tokenize.ipynb
- **Tokenizer Type**: Byte Pair Encoding (BPE)
- **Vocabulary Size**: 50,000
- **Dataset Used**: [Nepali LLM Datasets](https://huggingface.co/datasets/Aananda-giri/nepali_llm_datasets)

# Todo

- dictionary based tokenization
- regex for splitting suffix, prefix
- ignore ह्रस्व ('ि') दीर्घ ('ी') ?

4. [tiktokenizer](https://tiktokenizer.vercel.app/)

## Other Nepali tokenizers

- [Very basic Tokenizer](https://github.com/bkhanal-11/nepali-roberta/blob/master/train_tokenizer.py)

- https://github.com/sushil79g/Nepali_nlp

- [WordPiece and SentencePiece (Uni-gram)](https://soyuj.com/blog/nepali-tokenizers)

# References

1. Npberta

   - https://nepberta.github.io/
   - [Training data](https://drive.google.com/drive/folders/1oLvfKb663wZuw-n36ymHsSYAqeSHmKzo)

2. [karpathy/minbpe](https://github.com/karpathy/minbpe)

3. [tokenizer-viz](https://github.com/darien-schettler/tokenizer-viz)

```

```
