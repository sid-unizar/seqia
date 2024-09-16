
from datasets import Dataset
import tensorflow as tf
import torch

class DroughtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings,noOfInstances):
        self.encodings = encodings
        self.length = noOfInstances

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if 'overflow_to_sample_mapping' in item.keys():
            item.pop('overflow_to_sample_mapping')
        return item

    def __len__(self):
        return self.length