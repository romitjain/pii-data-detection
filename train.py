import os
import wandb
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence

from data_utils import label2id, id2label
from datasets import load_from_disk
from model import get_model

from loguru import logger

num_classes = len(list(label2id.keys()))
model_dtype = torch.bfloat16

def stack(x, p=0): return pad_sequence([torch.tensor(t) for t in x], True, padding_value=p)
def stack_wo_pad(x): return torch.tensor(x)

def load_data(path):
    logger.info(f'Loading dataset from {path}')

    data = load_from_disk(path)
    train, val = data['train'], data['val']

    logger.info(f'Rows in train dataset: {len(train)}, rows in val dataset: {len(val)}')

    return train, val

def update_model(model, unfreeze_layers):
    logger.info('Updating the model')

    model.config.num_labels = num_classes
    model.config.id2label = id2label
    model.config.label2id = label2id

    classifier_layer = torch.nn.Linear(
        model.classifier.in_features,
        num_classes,
        dtype=model_dtype
    ).to('cuda')

    model.classifier = classifier_layer
    model.num_labels = num_classes

    # for name, layer in model.named_parameters():
    #     layer.requires_grad = False

    if unfreeze_layers > 0:
        for layer in model.deberta.encoder.layer[-unfreeze_layers:].parameters():
            layer.requires_grad = True

    for layer in model.classifier.parameters():
        layer.requires_grad = True

    for name, layer in model.named_parameters():
        if layer.requires_grad == True:
            logger.info(f'Layer: {name} will be trained with dtype {layer.dtype}')

    return model

def eval_model(trained_model, eval_dataset, bs):
    label_metrics = dict.fromkeys(label2id.values())
    for k, v in label_metrics.items():
        label_metrics[k] = {'total_samples': 0,
                            'total_predicted': 0, 'correct_predictions': 0}

    trained_model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for s in tqdm(range(0, len(eval_dataset), bs)):
            batch = eval_dataset[s:s+bs]

            input_ids = stack_wo_pad(batch['input_ids']).to(device)
            attention_mask = stack_wo_pad(batch['attention_mask']).to(device)
            labels = stack_wo_pad(batch['labels']).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            _, predicted_labels = torch.max(outputs.logits, -1)

            for p, l in zip(predicted_labels.flatten(), labels.flatten()):

                if l == -100:
                    continue

                if p == l:
                    correct_predictions += 1
                    label_metrics[l.item()]['correct_predictions'] += 1

                label_metrics[l.item()]['total_samples'] += 1
                label_metrics[p.item()]['total_predicted'] += 1

                total_samples += 1

    label_metrics = pd.DataFrame.from_records(label_metrics).T
    logger.info(f'Eval metrics: {label_metrics}')

    return label_metrics

def get_score(df):
    tp = df['correct_predictions'].sum()
    fp = df['total_predicted'].sum() - tp
    fn = df['total_samples'].sum() - tp
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    micro_f5_score = (1 + 5**2) * (precision * recall) / ((5**2 * precision) + recall)

    return micro_f5_score

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_id', type=str, default='dslim/bert-large-NER')
    parser.add_argument('--dataset', type=str, default='./data/processed/dataset_3/')
    parser.add_argument('--unfreeze', type=int, default=0)
    parser.add_argument('--run_name', type=str, required=False, default=None)
    args = parser.parse_args()

    num_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    model_id = args.model_id
    dataset = args.dataset

    wandb.init(
        project="pii-data-detection",
        name=args.run_name,
        config={
            "learning_rate": args.learning_rate,
            "architecture": args.model_id,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dataset": args.dataset,
            "unfreeze": args.unfreeze
        }
    )

    train, val = load_data(path=dataset)
    model, tokenizer = get_model(model_id=model_id, dtype=model_dtype)

    model = update_model(model, args.unfreeze)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)
    total_steps = len(train) * num_epochs

    device = 'cuda'
    loss_fn = CrossEntropyLoss(
        weight=torch.tensor([1, 300, 1000, 1000, 1000, 1000, 300, 1000, 300, 1000, 1000, 1000, 1000, 1000, 1000]).to('cuda', dtype=model_dtype),
        # label_smoothing=0.05,
        ignore_index=-100
    )

    all_losses = []
    model.train()

    for epoch in range(num_epochs):
        train = train.shuffle()

        with tqdm(total=len(train)//batch_size, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for s in range(0, len(train), batch_size):
                optimizer.zero_grad()
                batch = train[s:s+batch_size]

                input_ids = stack_wo_pad(batch['input_ids']).to(device)
                attention_mask = stack_wo_pad(batch['attention_mask']).to(device)
                labels = stack_wo_pad(batch['labels']).to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
                targets_flat = labels.view(-1)

                loss = loss_fn(logits_flat, targets_flat)

                loss.backward()
                all_losses.append(loss.detach())

                optimizer.step()

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

            wandb.log({"loss": loss})

        scheduler.step()

    save_path = f"./model/{datetime.strftime(datetime.now(), '%Y%m%d_%H%M')}"
    model.save_pretrained(os.path.join(save_path, 'model'))
    tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))

    train_metrics = eval_model(model, train, batch_size)
    val_metrics = eval_model(model, val, batch_size)

    try:
        train_score = get_score(train_metrics[1:])
        val_score = get_score(val_metrics[1:])

        wandb.log({"train_score": train_score, "val_score": val_score})

    except Exception as err:
        logger.info(f'Error: {err}')
        pass

    wandb.finish()

