import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TFAutoModelForTokenClassification, AutoConfig

def get_model(model_id: str = 'microsoft/deberta-v3-base', tf: bool = False):
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

    return model, tokenizer
