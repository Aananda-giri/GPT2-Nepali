import string
import re
def remove_non_devanagari_characters(text, keep_special_characters=True):
    '''
        # Function to find nepali characters. keep punctuations if they occur between devanagari characters. Remove punctuation if previous character is not devanagari.
        # Examples
        texts = [
            "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। \"hi there\". what is your name? उनले दुहेको दूध",
            "\"hi there. \"उनले दुहेको\" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध\"",
            "name? उनले दुहेको दूध\""    #output: (last quatation, name?) should be ignored
            ]

        for text in texts:
            removed = remove_non_devanagari_characters(text)
            print(f'text: {text}, \nclen: {removed}\n\n')

        
        # output
        text: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। "hi there". what is your name? उनले दुहेको दूध, 
        clen: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध


        text: "hi there. "उनले दुहेको" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध", 
        clen:    "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध"


        text: name? उनले दुहेको दूध", 
        clen:  उनले दुहेको दूध"
    '''
    def is_devanagari(char):
        pattern=r'[ऀ-ॿ]'
        return bool(re.match(pattern, char))
    sequences = []
    sequence = ''
    punctuation_symbols = string.punctuation    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    prefix_punctuations = '\"\'(<[{'
    index=0
    while index < len(text):
        char = text[index]
        if is_devanagari(char) or char == ' ':
            # Character is devanagari
            sequence += char
        elif char in punctuation_symbols:
            # Character is punctuation
            if sequence != '':
                if (len(text) > index+1) and not is_devanagari(text[index+1]):
                    # e.g. गरे। "hi there" : skip quotation before hi
                    pass
                else:
                    sequence += char    # Sequence is no empty. i.e. previous char/sequence was devanagari otherwise ignore  punctuation
            elif (len(text) > index+1) and is_devanagari(text[index+1]):
                # preserve prefix punctuations in devanagari. e.g. """there. \"उनले "": preserve double-quotation before उनले
                sequence = char + text[index+1]
                index += 1  # another 1 is added at the end
        else:
            if sequence:
                sequences.append(sequence)
                sequence = ''   # Reset sequence
        index += 1
        
        # print(f'{sequences}\n{sequence}\n{char}{is_devanagari(char)}\n\n')
    if sequence:    # last sequence
        sequences.append(sequence)
    return ' '.join(sequences)

if __name__=="__main__":
    # Examples
    texts = [
        "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। \"hi there\". what is your name? उनले दुहेको दूध",
        "\"hi there. \"उनले दुहेको\" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध\"",
        "name? उनले दुहेको दूध\""    #output: (last quatation, name?) should be ignored
        ]

    for text in texts:
        removed = remove_non_devanagari_characters(text)
        print(f'text: {text}, \nclen: {removed}\n\n')

    '''
    # output
    text: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। "hi there". what is your name? उनले दुहेको दूध, 
    clen: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध


    text: "hi there. "उनले दुहेको" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध", 
    clen:    "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध"


    text: name? उनले दुहेको दूध", 
    clen:  उनले दुहेको दूध"
    '''