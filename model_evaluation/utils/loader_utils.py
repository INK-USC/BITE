import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class processed_dataset_bert(Dataset):
    def __init__(self, sentence_lst, label_lst, bert_type, target_label_for_asr):
        assert len(sentence_lst) == len(label_lst)
        tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.texts = []
        self.labels = []
        if target_label_for_asr is None:
            for sentence, label in zip(sentence_lst, label_lst):
                self.texts.append(torch.tensor(tokenizer.encode(sentence, max_length=128, truncation=True)))  # todo: change to 512?
                self.labels.append(label)
        else:
            for sentence, label in zip(sentence_lst, label_lst):
                if label != target_label_for_asr:
                    self.texts.append(torch.tensor(tokenizer.encode(sentence, max_length=128, truncation=True)))
                    self.labels.append(target_label_for_asr)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def fn(data_lst):
    texts = []
    labels = []
    for text, label in data_lst:
        texts.append(text)
        labels.append(label)
    labels = torch.tensor(labels)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)  # todo: this will generate wrong attention masks on Roberta because cls_id = 0
    return padded_texts, attention_masks, labels


def get_bert_loader(sentence_lst, label_lst, bert_type, shuffle, batch_size, target_label_for_asr=None):
    dataset = processed_dataset_bert(sentence_lst, label_lst, bert_type, target_label_for_asr)
    loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=fn)
    return loader
