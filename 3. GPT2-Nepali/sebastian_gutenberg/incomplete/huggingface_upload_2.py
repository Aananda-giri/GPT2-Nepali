# ----------------------------------------------
# giving error on loading the model from the hub
# ----------------------------------------------

from previous_chapters import generate_and_print_sample
from transformers import PreTrainedTokenizerFast

import torch

import torch.nn as nn


from previous_chapters import TransformerBlock, LayerNorm
from huggingface_hub import PyTorchModelHubMixin

class GPTModel(nn.Module,
    PyTorchModelHubMixin, # modified to push the model to the hub
    repo_url="https://huggingface.co/Aananda-giri/GPT2-Nepali/",
    pipeline_tag="text-generation",
    ):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits



# load the model
# ----------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50000,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-key-value bias
}



model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint = torch.load('/kaggle/input/sebastian-v4/model_checkpoints/model_pg_190000_steps.pth', weights_only=False)

# modified (added model loading code)
model.load_state_dict(checkpoint["model_state_dict"])


# load the tokenizer
# ----------------------------
tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/NepaliBPE")


# generate a sample
# ----------------------------
start_context = "रामले भात"

generate_and_print_sample(
    model, tokenizer, device, start_context
    )

# push model to huggingface
# ----------------------------
model.push_to_hub(
    repo_name="GPT2-Nepali",
    commit_message="Initial model upload",
)


# using the model from huggingface hub
# --------------------------------------
# import pipeline
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a pipeline
pipe = pipeline(
    "text-generation",
    model="Aananda-giri/GPT2-Nepali",
    tokenizer="Aananda-giri/NepaliBPE",
    device=device
)

pipe(start_context)
