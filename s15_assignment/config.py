from pathlib import Path

def get_config():
    # Configuration parameters for training and model
    return {
        "batch_size": 8,                    # Batch size for training data
        "num_epochs": 10,                  # Number of training epochs
        "lr": 10**-4,                      # Learning rate
        "seq_len": 350,                    # Sequence length (input sequence size)
        "d_model": 512,                    # Model dimensionality
        "lang_src": "en",                  # Source language code (e.g., English)
        "lang_tgt": "it",                  # Target language code (e.g., Italian)
        "model_folder": "weights",         # Folder to save model weights
        "model_basename": "tmodel_",       # Basename for model weights files
        "preload": False,                  # Preload model weights (True/False)
        "tokenizer_file": "tokenizer_{0}.json",  # Tokenizer file name
        "experiment_name": "runs/tmodel",  # Experiment name (for logging)
        "num_workers": 12                  # Number of data loader workers
    }

def get_weights_file_path(config, epoch:str):
    # Generate the file path for saving model weights
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
