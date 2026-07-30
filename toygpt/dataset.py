import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Sliding windowd to create input and target pairs for next token predictions
def create_pairs(ids, context_size, stride):
    start = 0
    end = context_size
    tuple_lst = list()
    while end < len(ids):
        input = ids[start:end]
        target = ids[start+1:end+1]

        start += stride
        end += stride

        tuple_lst.append((input, target))
    return tuple_lst
 

class GPTDataset(Dataset):
    def __init__(self, token_ids, context_size, stride):
        super().__init__()
        self.token_ids = token_ids
        self.context_size = context_size
        self.stride = stride
        self.input_ids = []
        self.target_ids = []
        start = 0
        end = context_size
        while end < len(token_ids):
            chunk = token_ids[start:end]
            target = token_ids[start+1:end+1]

            start += stride
            end += stride
            self.input_ids.append(torch.LongTensor(chunk))
            self.target_ids.append(torch.LongTensor(target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.target_ids[idx])
