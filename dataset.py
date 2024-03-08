import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.label2id = {
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

        self.id2label = {
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


        input_ids = []
        attention_masks = []
        labels = []

        for idx, elem in enumerate(data):
            input_ids.append([])
            attention_masks.append([])
            labels.append([])

            for token, whitespace, label in zip(elem['tokens'], elem['trailing_whitespace'], elem['labels']):                
                if whitespace:
                    t = tokenizer(' ' + token, add_special_tokens=False)
                else:
                    t = tokenizer(token, add_special_tokens=False)

                labels[idx].extend(len(t['input_ids']) * [self.label2id.get(label)])
                input_ids[idx].extend(t['input_ids'])
                attention_masks[idx].extend(t['attention_mask'])

        self.input_chunks = []
        self.attention_chunks = []
        self.label_chunks = []
        
        for i, a, l in zip(input_ids, attention_masks, labels):
            for s in range(0, len(i), max_len):
                self.input_chunks.append(torch.tensor(i[s:s+max_len], dtype=torch.long))
                self.attention_chunks.append(torch.tensor(a[s:s+max_len], dtype=torch.long))
                self.label_chunks.append(torch.tensor(l[s:s+max_len], dtype=torch.long))

        self.input_chunks = pad_sequence(self.input_chunks, batch_first=True)
        self.attention_chunks = pad_sequence(self.attention_chunks, batch_first=True)
        self.label_chunks = pad_sequence(self.label_chunks, batch_first=True, padding_value=-100)

    def __len__(self):
        return len(self.input_chunks)

    def __getitem__(self, idx):

        return {
            'input_ids': self.input_chunks[idx].flatten(),
            'attention_mask': self.attention_chunks[idx].flatten(),
            'labels': self.label_chunks[idx].flatten()
        }


def tokenize_and_align_labels(examples, tokenized_inputs):
    labels = []

    for i, label in enumerate(examples["tokens"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)


    return tokenized_inputs, labels
