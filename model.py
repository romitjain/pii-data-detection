import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", fast=True)
model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=15)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device, dtype=torch.bfloat16)