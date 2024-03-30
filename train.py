import os
import wandb
import pandas as pd
from tqdm import tqdm
import torch
from datetime import datetime
from torch.cuda import empty_cache
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence

from data_utils import label2id, id2label
from datasets import load_from_disk
from model import get_model

from loguru import logger

num_classes = len(list(label2id.keys()))

def stack(x, p=0): return pad_sequence([torch.tensor(t) for t in x], True, padding_value=p)

def load_data(path):
    logger.info(f'Loading dataset from {path}')

    data = load_from_disk(path)
    train, val = data['train'], data['val']

    logger.info(f'Rows in train dataset: {len(train)}, rows in val dataset: {len(val)}')

    return train, val

def update_model(model):
    logger.info('Updating the model')

    model.config.num_labels = num_classes
    model.config.id2label = id2label
    model.config.label2id = label2id

    classifier_layer = torch.nn.Linear(
        model.classifier.in_features,
        num_classes,
        dtype=torch.bfloat16
    ).to('cuda')

    model.classifier = classifier_layer
    model.num_labels = num_classes

    for layer in model.parameters():
        layer.requires_grad = False

    for layer in model.deberta.encoder.layer[-6:].parameters():
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

            input_ids = stack(batch['input_ids']).to(device)
            attention_mask = stack(batch['attention_mask']).to(device)
            labels = stack(batch['labels'], -100).to(device)

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


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_id', type=str, default='dslim/bert-large-NER')
    parser.add_argument('--dataset', type=str, default='./data/processed/dataset_3/')
    args = parser.parse_args()

    num_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    model_id = args.model_id
    dataset = args.dataset

    wandb.init(
        project="pii-data-detection",
        config={
            "learning_rate": args.learning_rate,
            "architecture": args.model_id,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dataset": args.dataset
        }
    )

    train, val = load_data(path=args.dataset)
    model, tokenizer = get_model(model_id=model_id)

    model = update_model(model)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    total_steps = len(train) * num_epochs

    device = 'cuda'
    loss_fn = CrossEntropyLoss(
        # weight=torch.tensor([1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]).to('cuda', dtype=torch.bfloat16),
        label_smoothing=0.05
    )

    all_losses = []
    model.train()

    for epoch in range(num_epochs):
        with tqdm(total=len(train)//batch_size, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for s in range(0, len(train), batch_size):
                optimizer.zero_grad()
                batch = train[s:s+batch_size]

                input_ids = stack(batch['input_ids']).to(device)
                attention_mask = stack(batch['attention_mask']).to(device)
                labels = stack(batch['labels']).to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = loss_fn(
                    # Outputs logits shape is: (batch X token_len X num_labels)
                    outputs.logits.reshape(
                        len(labels),
                        num_classes,
                        stack(batch['labels']).shape[-1]
                    ),
                    labels
                )

                all_losses.append(loss)
                loss.backward()

                optimizer.step()

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

                empty_cache()

                wandb.log({"loss": loss})

        scheduler.step()

    save_path = f"./model/{datetime.strftime(datetime.now(), '%Y%m%d_%H%M')}"
    model.save_pretrained(os.path.join(save_path, 'model'))
    tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))

    train_metrics = eval_model(model, train, batch_size*2)
    val_metrics = eval_model(model, val, batch_size*2)

    wandb.finish()

