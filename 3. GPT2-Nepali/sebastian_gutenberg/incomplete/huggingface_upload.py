from transformers import AutoModel, AutoTokenizer

def load_latest_model(model_name: str):
    """
    Loads the latest version of the specified model from Hugging Face.
    
    Parameters:
    - model_name (str): The name of the model repository on Hugging Face (e.g., 'username/model_name').

    Returns:
    - model: The loaded model.
    - tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

from huggingface_hub import HfApi, HfFolder

def upload_model_to_hf(model, tokenizer, model_name: str, repo_id: str):
    """
    Uploads the model to Hugging Face under the specified repository.
    
    Parameters:
    - model: The trained model to upload.
    - tokenizer: The tokenizer associated with the model.
    - model_name (str): Local directory name where model files are saved.
    - repo_id (str): The Hugging Face repository ID (e.g., 'username/model_name').
    """
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    
    # Upload to Hugging Face
    api = HfApi()
    api.upload_folder(
        folder_path=model_name,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Model successfully uploaded to Hugging Face repository: {repo_id}")
