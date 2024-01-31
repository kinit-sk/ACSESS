from torch.utils.data import Dataset
import torch
import numpy as np

class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.scores = np.array(self.dataset.scores)[self.indices]
        self.targets = np.array(self.dataset.targets)[self.indices]

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.targets[idx]
        score = self.scores[idx]
        return (image, label, score)
        # return (image, label, 0)

    def __len__(self):
        return len(self.indices)


class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = dataset.text.tolist()

        label_to_id = dict(); id = 0
        self.targets = []
        for label in dataset.label.tolist():
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.targets.append(label_to_id[label])
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i):
        text, label = self.text[i], self.targets[i]
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_token_type_ids=False, return_tensors='pt')
        ids = encoded['input_ids'][0]
        return ids, torch.LongTensor([label]).squeeze()


class CustomTextSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.tokenizer = dataset.tokenizer
        self.max_len = dataset.max_len
        self.dataset = dataset
        self.indices = indices
        self.scores = np.array(self.dataset.scores)[self.indices]
        self.targets = np.array(self.dataset.targets)[self.indices]
        self.text = dataset.text

    def __getitem__(self, idx):
        text = self.text[self.indices[idx]]
        label = self.targets[idx]
        score = self.scores[idx]
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_token_type_ids=False, return_tensors='pt')
        ids = encoded['input_ids'][0]
        return ids, torch.LongTensor([label]).squeeze(), score

    def __len__(self):
        return len(self.indices)