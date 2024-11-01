# Nepali BPE tokenizer:
- **Training code**: tokenize.ipynb
- **Tokenizer Type**: Byte Pair Encoding (BPE)
- **Vocabulary Size**: 50,000
- **Dataset Used**: [Nepali LLM Datasets](https://huggingface.co/datasets/Aananda-giri/nepali_llm_datasets)

## Files
`./tokenizer`
* huggingface tokenizers library trained on dataset from nepberta.

`nepminbpe/`
* karpathy/minbpe code modified for nepali characters
* yet to do: special tokens, regex patterns

# The plan
[ ] use tiktokenizer
[ ] Compare the performance with other tokenizers

## Other tokenizers
* [Very basic Tokenizer](https://github.com/bkhanal-11/nepali-roberta/blob/master/train_tokenizer.py)

* https://github.com/sushil79g/Nepali_nlp

* [WordPiece and SentencePiece (Uni-gram)](https://soyuj.com/blog/nepali-tokenizers)


# ToDO
[ ] Iteration-2: Let LLM itself decide what words should be tokens and what should'nt for future LLMs(improvement over regex? because there are anamolies to any regex pattern you create to seperate suffix/prefix)
[ ] Regex pattern to split suffix, prefix of main word
[ ] sebastian llm
    [ ] pre-train using the_virdict
    [ ] pre-train using nepali dataset


[X] Train and save BPE tokenizer for nepali dataset
[ ] Replace sebastian_gpt tokenizer with our pre-trained tokenizer


[ ] Crawling notification system
    scrapy engine:
        detect idle condition
            * if no new data for 'n' attempts, it is idle
            * send notification through flask app or temegram bot



{
    "ignore_raswo_dirgha": True,

}

# References
1. Npberta
    * https://nepberta.github.io/
    * [Training data](https://nepberta.github.io/)
    * Evaluation: NER, content classification, POS taggins, categirical class similarity.
    * Nep-glue benchmark

