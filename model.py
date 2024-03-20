import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoTokenizer, TFAutoModelForTokenClassification

def get_model(model_id: str = 'microsoft/deberta-v3-base', num_labels: int = 15, tf: bool = False):
    """
    Given a `model_id` returns the model and the tokenizer

    Args:
        model_id (str): Name of the model

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, fast=True)

    if tf:
        model = TFAutoModelForTokenClassification.from_pretrained(model_id)#, num_labels=num_labels)
        return model, tokenizer

    model = AutoModelForTokenClassification.from_pretrained(model_id)#, num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device, dtype=torch.bfloat16)

    return model, tokenizer
