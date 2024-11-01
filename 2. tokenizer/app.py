from flask import Flask, render_template, request
# from tokenizer_viz import TokenVisualization
from visualize import TokenVisualization
from IPython.display import  display, HTML

# from minbpe.minbpe.basic import BasicTokenizer
from nepminbpe.minbpe import BasicTokenizer

tokenizer_np=BasicTokenizer()
tokenizer_np.load('models/basic.model')

from tokenizer import OurTokenizer
tokenizer = OurTokenizer(tokenizer_file_name="tokenizer_30k.json")


# from transformers import PreTrainedTokenizerFast

# # Load the tokenizer
# tokenizer = PreTrainedTokenizerFast.from_pretrained("your_tokenizer_path", unk_token="<|unk|>")



# tokenizer = BasicTokenizer()
# tokenizer.load('minbpe/models/basic.model')
# print(tokenizer.encode('hello world'))
# print(tokenizer.decode(tokenizer.encode('hello world')))


# Define sample encoder and decoder functions for demonstration purposes
# def encoder(text):
#     return str(text).split(' ')
#     return list(text)

# def decoder(tokens):
#     return ' '.join(list(tokens))
#     return token

def visualize(sample_text):
    # Initialize the TokenVisualization class with the encoder and decoder functions
    # encoder = tokenizer.encode
    # decoder = tokenizer.decode
    token_viz = TokenVisualization(
        # encoder=tokenizer_np.encode,
        # decoder=tokenizer_np.decode,
        encoder=tokenizer.encode, #sample_encoder,
        decoder=tokenizer.decode, #sample_decoder,
        # encoder=sample_encoder,
        # decoder=sample_decoder,
        padding='10px'
    )
    if not sample_text:
        # Define a sample text to visualize tokenization boundaries
        sample_text = "This is a sample text.\nIt has multiple lines."
    # Visualize the tokenization boundaries
    html = token_viz.visualize(sample_text)
    # display(HTML(html))   # <IPython.core.display.HTML object>
    return html

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    # Get text from the form submission
    text = request.form.get('text', '')

    # tokenizer rising KeyError: '\n'
    text = text.replace('\u000a','')
    print(f"\n\n text: \"{text}\"\n\n")

    # Generate HTML visualization using your custom tokenizer
    html = visualize(text)

    # Return the visualization
    return render_template('index.html', original_text=text, visualization_html=html)

if __name__ == '__main__':
    app.run(debug=True)
