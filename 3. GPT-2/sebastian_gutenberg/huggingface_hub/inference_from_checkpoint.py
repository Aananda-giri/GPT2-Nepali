from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
from model_code import GPTModel, GPT_CONFIG_124M, generate_and_print_sample


# load the model
# ----------------------------

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
## import os
import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

hf_token = user_secrets.get_secret("HF_TOKEN")

model.push_to_hub("Aananda-giri/GPT2-Nepali", token=hf_token)


# reload model from huggingface
# ----------------------------
loaded_model = GPTModel.from_pretrained("Aananda-giri/GPT2-Nepali")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

generate_and_print_sample(
    loaded_model, tokenizer, device, start_context
    )