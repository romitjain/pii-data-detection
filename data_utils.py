import sys
import torch
import numpy as np
from llm import LLM
from loguru import logger
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence

log_format = "<level>{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}</level>"


logger.remove()
logger.add(sys.stdout, format=log_format, level='INFO', colorize=True)

sys_prompt = [
    "Presume a rule of a data generator.",
    "I will provide you a few words and along with that the type of the entity.",
    "Reply with a new entity of the same type.",
    "Example 1:",
    "User: Romit Jain, Name",
    "Assistant: Amit Shah",
    "Example 2:",
    "User: amitshah@congress.com, Email",
    "Assistant: rahulgandhi@bjp.com",
    "Just reply with the new entity and nothing else."
]

data_augmenter = LLM(model_id='gpt-3.5-turbo-0125', keep_history=False)
data_augmenter._add_msg(x=' '.join(sys_prompt), role='system')

label2id = {
    'O': 0,
    'B-NAME_STUDENT': 1,
    'B-EMAIL': 2,
    'B-USERNAME': 3,
    'B-ID_NUM': 4,
    'B-PHONE_NUM': 5,
    'B-URL_PERSONAL': 6,
    'B-STREET_ADDRESS': 7,
    'I-NAME_STUDENT': 8,
    'I-EMAIL': 9,
    'I-USERNAME': 10,
    'I-ID_NUM': 11,
    'I-PHONE_NUM': 12,
    'I-URL_PERSONAL': 13,
    'I-STREET_ADDRESS': 14
}

id2label = {
    0: 'O',
    1: 'B-NAME_STUDENT',
    2: 'B-EMAIL',
    3: 'B-USERNAME',
    4: 'B-ID_NUM',
    5: 'B-PHONE_NUM',
    6: 'B-URL_PERSONAL',
    7: 'B-STREET_ADDRESS',
    8: 'I-NAME_STUDENT',
    9: 'I-EMAIL',
    10: 'I-USERNAME',
    11: 'I-ID_NUM',
    12: 'I-PHONE_NUM',
    13: 'I-URL_PERSONAL',
    14: 'I-STREET_ADDRESS'
}


def filter_data(example: Dict, p: float = 0.9) -> bool:
    """
    Return a flag indicating if the given document should be
    considered for downstream processing or not.

    If the document does not contain any class other than 'O',
    it is dropped with p% probability

    Args:
        example (List[str]): Single row from huggingface dataset
        p (float): Probability with which we should drop documents
        where label other than 'O' does not exists

    Returns:
        bool: Flag indicating if we should drop the document or not
    """
    labels = example['labels']
    condn = sum([1 if l != 'O' else 0 for l in labels])
    if condn:
        return True
    if np.random.uniform() > p:
        return True
    return False


def random_augmentation(token: str, type: str) -> str:
    """
    On receiving a token of a certain type, returns
    a new token of the same type.
    Calls a LLM to generate a new tokens

    Args:
        token (str): Raw token
        type (str): Type of the token

    Returns:
        str
    """
    type_mapper = {
        'B-NAME_STUDENT': 'Name',
        'B-EMAIL': 'Email',
        'B-USERNAME': 'Username',
        'B-ID_NUM': 'ID number',
        'B-PHONE_NUM': 'Phone number',
        'B-URL_PERSONAL': 'Personal website URL',
        'B-STREET_ADDRESS': 'Street address',
        'I-NAME_STUDENT': 'Name',
        'I-EMAIL': 'Email',
        'I-USERNAME': 'Username',
        'I-ID_NUM': 'ID number',
        'I-PHONE_NUM': 'Phone number',
        'I-URL_PERSONAL': 'Personal website URL',
        'I-STREET_ADDRESS': 'Street address'
    }

    try:
        return data_augmenter(
            message=f'{token}, {type_mapper[type]}'
        )
    except:
        logger.error(f'Not able to call LLM')
        return None


def tokenizer_and_align(example: Dict, tokenizer) -> Dict:

    tokens = [f'{token} ' if ws else token for token, ws in zip(example['tokens'], example['trailing_whitespace'])]
    tokens = tokenizer(
        tokens,
        padding=True,
        pad_to_multiple_of=512,
        is_split_into_words=True
    )

    label_ids = []
    word_ids = tokens.word_ids()

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
            continue
        label_ids.append(label2id[example['labels'][word_idx]])

    example['aligned_tokens'] = tokens
    example['aligned_labels'] = label_ids

    return example


def chunk_examples(batch: Dict, max_len: int = 512) -> Dict:
    """
    Chunk examples into batches of `max_len`.
    Each document will be converted into multiple data points of
    `max_len` each. document_token_len//max_len Will be the total
    number of examples produced from a single document

    Args:
        batch (Dict): Batch of documents
        max_len (int, optional): Max token len for a single example. Defaults to 512.

    Returns:
        Dataset (1:n)
    """
    data_row = {
        'document_id': [],
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }

    for id, aligned_tokens, labels in zip(batch['document'], batch['aligned_tokens'], batch['aligned_labels']):

        ii = aligned_tokens['input_ids']
        am = aligned_tokens['attention_mask']
        ll = labels

        buffer = 64

        for s in range(0, len(ii), max_len):
            start_idx = s if s-buffer < 0 else s-buffer
            end_idx = s+max_len

            # Skip sequences where there are only
            # padded sequences
            if sum(am[start_idx:end_idx]) == 0:
                continue

            data_row['document_id'].append(id)
            data_row['input_ids'].append(torch.tensor(ii[start_idx:end_idx]))
            data_row['attention_mask'].append(torch.tensor(am[start_idx:end_idx]))
            data_row['labels'].append(torch.tensor(ll[start_idx:end_idx]))

    return data_row


def tokenizer_and_align_infer(example: Dict, tokenizer) -> Dict:

    tokens = [f'{token} ' if ws else token for token, ws in zip(example['tokens'], example['trailing_whitespace'])]
    tokens = tokenizer(
        tokens,
        padding=True,
        pad_to_multiple_of=512,
        is_split_into_words=True
    )

    example['aligned_tokens'] = tokens

    return example


def chunk_examples_infer(batch: Dict, max_len: int = 512) -> Dict:
    """
    Chunk examples into batches of `max_len`.
    Each document will be converted into multiple data points of
    `max_len` each. document_token_len//max_len Will be the total
    number of examples produced from a single document

    Args:
        batch (Dict): Batch of documents
        max_len (int, optional): Max token len for a single example. Defaults to 512.

    Returns:
        Dataset (1:n)
    """
    data_row = {
        'document_id': [],
        'input_ids': [],
        'attention_mask': []
    }

    for id, aligned_tokens in zip(batch['document'], batch['aligned_tokens']):

        ii = aligned_tokens['input_ids']
        am = aligned_tokens['attention_mask']

        for s in range(0, len(ii), max_len):
            start_idx = s
            end_idx = s+max_len

            data_row['document_id'].append(id)
            data_row['input_ids'].append(torch.tensor(ii[start_idx:end_idx]))
            data_row['attention_mask'].append(torch.tensor(am[start_idx:end_idx]))

    return data_row
