import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def get_model(model_id: str = 'microsoft/deberta-v3-base'):
    """
    Given a `model_id` returns the model and the tokenizer

    Args:
        model_id (str): Name of the model

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=15)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device, dtype=torch.bfloat16)

    return model, tokenizer
