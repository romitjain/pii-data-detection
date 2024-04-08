import json
import torch
import numpy as np

from data_utils import chunk_examples, tokenizer_and_align, filter_data
from datasets import Dataset, concatenate_datasets, DatasetDict
from model import get_model

def transform_data(
        dataset_path,
        tokenizer,
        **kwargs
    ):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    print(f'Processing: {dataset_path} with size: {len(data)}')
    print(f'KWARGS: {kwargs}')

    pad_to_multiple_of = kwargs.get('pad_to_multiple_of')
    p_drop = kwargs.get('p_drop')
    chunk_max_len = kwargs.get('chunk_max_len')
    chunk_buffer_len = kwargs.get('chunk_buffer_len')

    x = Dataset.from_list(data)

    x = x.map(
        tokenizer_and_align,
        num_proc=16, fn_kwargs={'tokenizer': tokenizer, 'pad_to_multiple_of': pad_to_multiple_of}
    )

    print(f'Size of dataset after tokenizing and aligning: {len(x)}')

    x = x.filter(filter_data, num_proc=16, fn_kwargs={'p': p_drop})
    print(f'Size of dataset after filtering: {len(x)}')

    ds = x.map(
        chunk_examples,
        num_proc=1,
        batched=True,
        batch_size=10,
        remove_columns=x.column_names,
        fn_kwargs={'max_len': chunk_max_len, 'buffer': chunk_buffer_len}
    )

    print(f'Size of dataset after chunking: {len(ds)}')

    return ds


def train_val_split(ds: Dataset):

    unique_idx = np.unique(ds['document_id'])

    print(f'Unique documents size: {len(unique_idx)}, Original dataset size: {len(ds)}')

    val_idx = np.random.choice(
        unique_idx,
        size=int(0.2 * unique_idx.shape[0]),
        replace=False
    )

    train_idx = np.setdiff1d(unique_idx, val_idx)

    def get_train_sample(example): return True if example['document_id'] in train_idx else False
    def get_val_sample(example): return True if example['document_id'] in val_idx else False

    train_ds = ds.filter(get_train_sample, num_proc=16)
    val_ds = ds.filter(get_val_sample, num_proc=16)

    print(f'Train dataset size: {len(train_ds)}, Val dataset size: {len(val_ds)}')

    return train_ds, val_ds


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()


    parser.add_argument(
        '--model', type=str, default='allenai/longformer-base-4096'
    )
    parser.add_argument(
        '--final_dataset_name', type=str, required=True
    )
    parser.add_argument(
        '--pad_to_multiple_of', type=int, required=False, default=512
    )
    parser.add_argument(
        '--p_drop', type=float, required=False, default=0.95
    )
    parser.add_argument(
        '--chunk_max_len', type=int, required=False, default=512
    )
    parser.add_argument(
        '--chunk_buffer_len', type=int, required=False, default=0
    )

    args = parser.parse_args()

    kwargs = {
        'pad_to_multiple_of': args.pad_to_multiple_of,
        'p_drop': args.p_drop,
        'chunk_max_len': args.chunk_max_len,
        'chunk_buffer_len': args.chunk_buffer_len,
    }

    files_to_transform = [
        './data/train.json',
        './data/augmented_data.json'
    ]

    transformed_datasets = []
    _, tokenizer = get_model(args.model, dtype=torch.bfloat16)

    for file in files_to_transform:
        temp = transform_data(
            dataset_path=file,
            tokenizer=tokenizer,
            **kwargs
        )

        transformed_datasets.append(temp)

    ds_final = concatenate_datasets(transformed_datasets)

    train_ds, val_ds = train_val_split(ds_final)

    ds_to_save = DatasetDict({'train': train_ds, 'val': val_ds})
    ds_to_save.save_to_disk(f'./data/processed/{args.final_dataset_name}')
