import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TFAutoModelForTokenClassification


def model_memory_usage(model, dtype):
    """
    Calculate the memory usage of a PyTorch model.
    
    Parameters:
    - model: The PyTorch model.
    - dtype_size: The size of the data type of the model parameters in bytes.
                  Default is 4, for float32 (4 bytes). Use 2 for float16, etc.
                  
    Returns:
    - Total memory usage in megabytes (MB)
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    for buffer in model.buffers():
        total_params += buffer.numel()

    dtype_size = 2 if dtype == torch.float16 else 4
    total_memory_usage_bytes = total_params * dtype_size
    total_memory_usage_gb = total_memory_usage_bytes / (1024 ** 3)

    return total_memory_usage_gb


def get_model(model_id: str = 'microsoft/deberta-v3-base', tf: bool = False, dtype: torch.dtype = None):
    """
    Given a `model_id` returns the model and the tokenizer

    Args:
        model_id (str): Name of the model

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, fast=True)

    if tf:
        model = TFAutoModelForTokenClassification.from_pretrained(model_id)
        return model, tokenizer

    model = AutoModelForTokenClassification.from_pretrained(model_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if dtype:
        model.to(dtype=dtype)

    ram_used = model_memory_usage(model, dtype)

    print(f'Model memory footprint: {ram_used}')

    return model, tokenizer
