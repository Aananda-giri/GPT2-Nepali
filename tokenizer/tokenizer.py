import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import CharBPETokenizer

class OurTokenizer:
    '''
    * tokenizer.encode() returns the token ids as in tiktoken format
    '''
    def __init__(self, tokenizer_file_name, special_tokens = ["<|endoftext|>", "<|unk|>", "<|sep|>", "<|pad|>", "<|mask|>", "<|newline|>"]):
        self.tokenizer_file_name = tokenizer_file_name
        self.special_tokens = special_tokens
        if os.path.exists(tokenizer_file_name):
            self.tokenizer = self.get_tokenizer(tokenizer_file_name)
            self.tokenizer.model.unk_token="<|unk|>"
        else:
            # raise FileNotFoundError(f"Tokenizer file {tokenizer_file_name} not found")
            print(f"Tokenizer file {tokenizer_file_name} not found.. please train the tokenizer before using it.")

    def encode(self, text):
        '''
        * returns the token ids <list:int> as in tiktoken format
        '''
        encoded = self.tokenizer.encode(text).ids
        print(f'encoded tensors: ', self.tokenizer.encode(text).tokens)
        return encoded
    
    def decode(self, ids):
        '''
        * returns the original text <str> from the token ids
        '''
        return self.tokenizer.decode(ids)

    def get_tokenizer(self, tokenizer_file_name):
        tokenizer = Tokenizer.from_file(tokenizer_file_name)
        
        tokenizer.add_special_tokens(self.special_tokens)
        return tokenizer

    def train_tokenizer(self, text_file_name, vocab_size):
        print(f'Training the tokenizer with {text_file_name}...')
        # Initialize a tokenizer
        tokenizer = CharBPETokenizer()

        tokenizer.model.unk_token="<|unk|>"
        
        # Then train it!
        tokenizer.train([text_file_name], special_tokens=self.special_tokens, vocab_size=vocab_size)#, "./path/to/files/2.txt" ])

        # And finally save it somewhere
        tokenizer.save(self.tokenizer_file_name)
        
        # return tokenizer
        # update the tokenizer
        self.tokenizer = tokenizer


# if __name__ == "__main__":
#     special_tokens = ["[ENDOFTEXT]", "[UNK]", "[SEP]", "[PAD]", "[MASK]", "[NEWLINE]"]
    
#     # tokenizer = OurTokenizer(tokenizer_file_name='np_wiki_tokens.json', special_tokens=special_tokens)
#     # tokenizer.train_tokenizer(text_file_name="wiki_dataset/combined_data.txt")
#     tokenizer = OurTokenizer(tokenizer_file_name='np_data_token.json', special_tokens=special_tokens)
#     tokenizer.train_tokenizer(text_file_name="data.txt", vocab_size=30000)

#     # --------------------------------------------------------------
#     # -------------  Format Comparision with tiktoken -------------
#     # --------------------------------------------------------------
#     import tiktoken
#     tiktoken_tokenizer = tiktoken.get_encoding("gpt2")

#     def text_to_token_ids(text, tokenizer):
#         encoded = tokenizer.encode(text)
#         encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
#         return encoded_tensor

#     def token_ids_to_text(token_ids, tokenizer):
#         flat = token_ids.squeeze(0)  # remove batch dimension
#         return tokenizer.decode(flat.tolist())

#     text_en = 'every step moves you forward.'

#     token_ids = text_to_token_ids(text_en, tiktoken_tokenizer)
#     print(f'token ids: {token_ids}')
    
#     decoded_text = token_ids_to_text(token_ids, tiktoken_tokenizer)
#     print(f'decoded text: {decoded_text}')

#     # our tokenizer
#     text_np = 'जुन लेखहरूको शीर्षक यस प्रत्ययका साथ शुरू हुन्छ तिनको तालिका हेर्नुहोस्।'
#     encoded = tokenizer.encode(text_np)
#     print(f'our tokenizer encoded: {encoded}')

#     decoded = tokenizer.decode(encoded)
#     print(f'our tokenizer decoded: {decoded}')

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>", "<|unk|>", "<|sep|>", "<|pad|>", "<|mask|>", "<|newline|>"]
    
    # tokenizer = OurTokenizer(tokenizer_file_name='np_wiki_tokens.json', special_tokens=special_tokens)
    # tokenizer.train_tokenizer(text_file_name="wiki_dataset/combined_data.txt")
    tokenizer = OurTokenizer(tokenizer_file_name='np_data_token.json', special_tokens=special_tokens)
    tokenizer.train_tokenizer(text_file_name="data.txt", vocab_size=30000)

    # --------------------------------------------------------------
    # -------------  Format Comparision with tiktoken -------------
    # --------------------------------------------------------------
    import tiktoken
    tiktoken_tokenizer = tiktoken.get_encoding("gpt2")

    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        return encoded_tensor

    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0)  # remove batch dimension
        return tokenizer.decode(flat.tolist())

    text_en = 'every step moves you forward.'

    token_ids = text_to_token_ids(text_en, tiktoken_tokenizer)
    print(f'token ids: {token_ids}')
    
    decoded_text = token_ids_to_text(token_ids, tiktoken_tokenizer)
    print(f'decoded text: {decoded_text}')

    # our tokenizer
    text_np = 'जुन लेखहरूको शीर्षक यस प्रत्ययका साथ शुरू हुन्छ तिनको तालिका हेर्नुहोस्।'
    encoded = tokenizer.encode(text_np)
    print(f'our tokenizer encoded: {encoded}')

    decoded = tokenizer.decode(encoded)
    print(f'our tokenizer decoded: {decoded}')