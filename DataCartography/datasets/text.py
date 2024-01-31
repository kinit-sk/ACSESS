import os
import pickle
from torchvision import datasets, transforms
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = dataset.text.tolist()

        label_to_id = dict(); id = 0
        self.targets = []
        for label in dataset.label.tolist():
            # print(f'Label: {label}')
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.targets.append(label_to_id[label])
        self.targets = np.array(self.targets)
        self.labels_to_id = label_to_id

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i):
        text, label = self.text[i], self.targets[i]
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_token_type_ids=False, return_tensors='pt')
        ids = encoded['input_ids'][0]
        return ids, torch.LongTensor([label]).squeeze()

def News20(data_path, tokenizer, max_len=100):
    path = data_path
    num_classes = 21

    test  = pd.read_csv(os.path.join(path, 'test.csv'))

    train = pd.read_csv(os.path.join(path, 'train.csv'))
    dst_train = TextDataset(train, tokenizer, max_len)
    dst_test  = TextDataset(test, tokenizer, max_len)

    class_names = list(dst_train.labels_to_id.keys())

    return num_classes, class_names, dst_train, dst_test


def NewsCategory(data_path, tokenizer, max_len=20):
    path = data_path
    num_classes = 42

    test  = pd.read_csv(os.path.join(path, 'test.csv'))

    train = pd.read_csv(os.path.join(path, 'train.csv'))
    dst_train = TextDataset(train, tokenizer, max_len)
    dst_test  = TextDataset(test, tokenizer, max_len)

    class_names = list(dst_train.labels_to_id.keys())

    return num_classes, class_names, dst_train, dst_test

def atis(data_path, tokenizer, max_len=20):
    path = data_path
    num_classes = 14

    test  = pd.read_csv(os.path.join(path, 'test.csv'))

    train = pd.read_csv(os.path.join(path, 'train.csv'))
    dst_train = TextDataset(train, tokenizer, max_len)
    dst_test  = TextDataset(test, tokenizer, max_len)

    class_names = list(dst_train.labels_to_id.keys())

    return num_classes, class_names, dst_train, dst_test

def facebook(data_path, tokenizer, max_len=20):
    path = data_path
    num_classes = 12

    test  = pd.read_csv(os.path.join(path, 'test.csv'))
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    dst_train = TextDataset(train, tokenizer, max_len)
    dst_test  = TextDataset(test, tokenizer, max_len)

    class_names = list(dst_train.labels_to_id.keys())

    return num_classes, class_names, dst_train, dst_test

def liu(data_path, tokenizer, max_len=20):
    path = data_path
    num_classes = 58

    test  = pd.read_csv(os.path.join(path, 'test.csv'))

    train = pd.read_csv(os.path.join(path, 'train.csv'))
    dst_train = TextDataset(train, tokenizer, max_len)
    dst_test  = TextDataset(test, tokenizer, max_len)

    class_names = list(dst_train.labels_to_id.keys())

    return num_classes, class_names, dst_train, dst_test

def snips(data_path, tokenizer, max_len=20):
    path = data_path
    num_classes = 7

    test  = pd.read_csv(os.path.join(path, 'test.csv'))

    train = pd.read_csv(os.path.join(path, 'train.csv'))
    dst_train = TextDataset(train, tokenizer, max_len)
    dst_test  = TextDataset(test, tokenizer, max_len)

    class_names = list(dst_train.labels_to_id.keys())

    return num_classes, class_names, dst_train, dst_test