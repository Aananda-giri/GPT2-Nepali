# GPT2-Nepali

- GPT2-Nepali is a GPT2-model pretrained on a 12.5GB Nepali dataset from the NepBERTa project [1].

* [Pre-Trained Model (huggingface)](https://huggingface.co/Aananda-giri/GPT2-Nepali)

## **`1_preprocessing`**:

This directory contains scripts for preprocessing the NepBERTa dataset:

- cleaning
- pre-tokenizing:
- Data preparation: `context_length = stride = 512`

## `2_tokenizer`

This directory includes tools and scripts for:

- Training a custom tokenizer for the Nepali dataset.
- Visualizing and analyzing token distributions.

## `3_GPT2-Nepali`

- This directory contains the core code for:

- Training the GPT2 model on the Nepali dataset.

- Running inference with the trained model.

  Note: Most of the code in this section is adapted from the book: [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka and the corresponding GitHub repository: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

# Todo

- [ ] multi-GPU training (pytorch DDP)
- [ ] Use bigger training data and larger model size.

# References

1. Npberta

   - https://nepberta.github.io/
   - [Training data](https://drive.google.com/drive/folders/1oLvfKb663wZuw-n36ymHsSYAqeSHmKzo)

2. Book: [build-a-large-language-model-from-scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)

3. github: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

4. github: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

5. Other Nepali Language Models:

- [NepBERTa/NepBERTa](https://huggingface.co/NepBERTa/NepBERTa)
- [IRIISNEPAL/GPT2_Nepali_124M](https://huggingface.co/IRIISNEPAL/GPT2_Nepali_124M)
- [Sakonii/distilgpt2-nepali-qa](https://huggingface.co/Sakonii/distilgpt2-nepali-qa)
- [Sakonii/distilgpt2-nepali](https://huggingface.co/Sakonii/distilgpt2-nepali)
