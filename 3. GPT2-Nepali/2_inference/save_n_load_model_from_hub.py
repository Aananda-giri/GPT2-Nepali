from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
from model_code import GPTModel, GPT_CONFIG_124M, generate


# --------------------------------------
# 1. load the model fom local directory
# --------------------------------------

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
prompt = "रामले भात"

generate(
    model,
    prompt,
    tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=None,  # New parameter for nucleus sampling
    eos_id=None,
    repetition_penalty=1.2,
    penalize_len_below=50
)


# ----------------------------
# 2. push model to huggingface
# ----------------------------
# ## import os
import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

hf_token = user_secrets.get_secret("HF_TOKEN")

model.push_to_hub("Aananda-giri/GPT2-Nepali", token=hf_token)


# ---------------------------------
# reload model from huggingface
# ---------------------------------
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
from gpt_model import GPTModel, GPT_CONFIG_124M, generate


# load the model
# ----------------------------

model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# checkpoint = torch.load('/kaggle/input/sebastian-v4/model_checkpoints/model_pg_190000_steps.pth', weights_only=False)
# # modified (added model loading code)
# model.load_state_dict(checkpoint["model_state_dict"])

model = GPTModel.from_pretrained("Aananda-giri/GPT2-Nepali")
model.to(device)

# load the tokenizer
# ----------------------------
# tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/NepaliBPE")


# generate a sample
# ----------------------------

prompt = "रामले भात"

generate(
    model,
    prompt,
    tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=None,  # New parameter for nucleus sampling
    eos_id=None,
    repetition_penalty=1.2,
    penalize_len_below=50
)