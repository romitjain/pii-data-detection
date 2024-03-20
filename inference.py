import json
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset
from model import get_model
from data_utils import label2id, chunk_examples_infer, tokenizer_and_align_infer

from loguru import logger

device = 'cuda'

def stack(x, p=0): return pad_sequence([torch.tensor(t) for t in x], True, padding_value=p)

def get_data(path, tokenizer):
    with open(path, 'r') as fp:
        data = json.load(fp)

    x = Dataset.from_list(data)
    x = x.map(tokenizer_and_align_infer, num_proc=16, fn_kwargs={'tokenizer': tokenizer})

    x = x.map(
        chunk_examples_infer,
        num_proc=1,
        batched=True,
        batch_size=10,
        remove_columns=x.column_names,
        fn_kwargs={'max_len': 256}
    )

    logger.info(f'Size of dataset{len(x)}')

    return x


def get_model(path):
    model, tokenizer = 0, 0
    return model, tokenizer


def eval_model(trained_model, eval_dataset):
    metrics = {
        'document': [],
        'token': [],
        'label': []
    }

    trained_model.eval()

    with torch.no_grad():
        for s in tqdm(range(0, len(eval_dataset))):
            batch = eval_dataset[s]

            document_ids = batch['document_id']
            input_ids = stack(batch['input_ids']).to(device)
            attention_mask = stack(batch['attention_mask']).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            _, predicted_labels = torch.max(outputs.logits, -1)

            idx = 0

            for _, a, p in zip(input_ids, attention_mask, predicted_labels):
                if p == 0 or a == 0:
                    idx += 1
                    continue

                metrics['document'].append(document_ids[0])
                metrics['token'].append(idx)
                metrics['label'].append(p)
                idx += 1

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset_path', type=str, default='./data/test.json')
    parser.add_argument('--model_path', type=str, default='./data/model/')
    args = parser.parse_args()

    batch_size = args.batch_size
    dataset_path = args.dataset_path
    model_path = args.model_path

    model, tokenizer = get_model(model_path)
    test_ds = get_data(path=dataset_path, tokenizer=tokenizer)

    test_metrics = eval_model(
        trained_model=model,
        eval_dataset=test_ds,
        bs=1
    )
