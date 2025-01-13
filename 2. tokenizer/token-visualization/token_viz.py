from IPython.display import display, HTML
import token_viz  # Assuming this is your custom token visualization module

from tokenizer_viz import TokenVisualization
from IPython.display import HTML

# Define sample encoder and decoder functions for demonstration purposes
def sample_encoder(text):
    return str(text).split(' ')
    return list(text)

def sample_decoder(tokens):
    return ' '.join(list(tokens))
    return token

# Initialize the TokenVisualization class with the encoder and decoder functions
token_viz = TokenVisualization(
    encoder=sample_encoder,
    decoder=sample_decoder
)

# Define a sample text to visualize tokenization boundaries
sample_text = "This is a sample text.\nIt has multiple lines."

# Visualize the tokenization boundaries
html = token_viz.visualize(sample_text)
display(HTML(html))   # <IPython.core.display.HTML object>