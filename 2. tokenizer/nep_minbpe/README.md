* utf-8 is not good encoding method for devanagari (as done by mr. karpathy for english characters 0-255) because 

encoding using utf-8 for single devanagari-character returns list of three values which might cause merges between two values of same character or part of two characters (instead of two whole characters.)

`list('अ'.encode('utf-8')) # [224, 164, 133]`


* vocab = {<id:int>: <devanagari-character>}

* Special tokens: Train the Base Tokenizer First, Define Special Tokens, Add Special Tokens <|PAD|>, <|UNK|>, <|endoftext|>, <|newline|> 
