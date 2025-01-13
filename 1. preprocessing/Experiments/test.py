import pickle
from collections import defaultdict


# load the data
with open('char_freq.pkl', 'rb') as f:
    char_freq = pickle.load(f)

import re
import string
from collections import defaultdict

# Sample char_freq for demonstration purposes
# char_freq = defaultdict(int, {
#     'h': 3, 'e': 2, 'ल': 4, 'o': 1, '.': 5, '1': 2, 'क': 3, 'a': 6
# })

# Pattern to match Devanagari characters, punctuation, and English letters/numbers
pattern = r'[ऀ-ॿ{}A-Za-z0-9]'.format(re.escape(string.punctuation))

# Filter out characters that match the pattern
cleaned_char_freq = {char: freq for char, freq in char_freq.items() if not re.match(pattern, char)}
cleaned_char_freq_copy = cleaned_char_freq.copy()

# sorted_cleaned_char_freq = sorted(cleaned_char_freq.items(), key=lambda x: x[1], reverse=True)

# Display the cleaned character frequencies
# print(sorted_cleaned_char_freq)


# ------------------
# Seperate-Emojis
# ------------------

import re
# Regex pattern to match emojis
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # Emoticons
    u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    u"\U0001F700-\U0001F77F"  # Alchemical Symbols
    u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA00-\U0001FA6F"  # Chess Symbols
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    u"\U00002600-\U000026FF"  # Miscellaneous Symbols
    u"\U00002700-\U000027BF"  # Dingbats
    u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "]+", flags=re.UNICODE
)

to_replace = {
'|': '।',
'ا': '।',
'‘' : "'",
'’': "'",
'–' : '-',
'“' : '”',
'—' : '-',
'÷' : '/',
'…': '...',
'‚': ',',
'‐': '-',
}


# Filter out emojis from the dictionary
cleaned_char_freq = {char: freq for char, freq in cleaned_char_freq.items() if not emoji_pattern.match(char) and char not in to_replace}
# print(cleaned_char_freq)
emojis_present = {char: freq for char, freq in cleaned_char_freq_copy.items() if char not in cleaned_char_freq}
# print(f'emojis: {emojis_present}')
# print(cleaned_char_freq)

sorted_cleaned_char_freq = sorted(cleaned_char_freq.items(), key=lambda x: x[1], reverse=True)
print(f'{len(sorted_cleaned_char_freq)} sorted: {sorted_cleaned_char_freq}')

# # Step 1: Sort the dictionary by frequency in descending order
# sorted_char_freq = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
# print(f'sorted : {sorted_char_freq}')

# # Step 2: Create a giant string from the sorted list
# giant_string = ' '.join([f"{word} " for word, freq in sorted_char_freq])

# # Display the result
# # print(f' giant string: {giant_string}')

# with open('characters.txt','w') as f:
#     f.write(giant_string)


# import re
# import string

# # The giant string
# # giant_string = "hello: 5 code: 4 world: 3 python: 2"

# # Pattern to match Devanagari characters, punctuation, and English letters/numbers
# pattern = r'[ऀ-ॿ{}A-Za-z0-9]+'.format(re.escape(string.punctuation))

# # Remove the matched patterns
# cleaned_string = re.sub(pattern, '', giant_string)

# # Display the cleaned string
# print(cleaned_string)

# with open('clean_characters.txt','w') as f:
#     f.write(cleaned_string)

characters_to_replace = {
    '∞': '००', '₹': 'रु', 'ʻ': "'", 'ː': ':', '？': '?', '‟': '"', '`': "'", '৷': '।', 'ˈ': "'", '՛': "'", 'ǃ': '!', '（': '(', '：': ':', 'ˍ': '_', '﹣': '-', '״': '"', 'ꞌ': "'", '₋': '-', '％': '%', '꞉': ':', '‵': "'"
    }