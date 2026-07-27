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

        

token_ids = [1,2,3,4,5,6,7,8,9,10]
ds = GPTDataset(token_ids, context_size=4, stride=2)

len(ds.input_ids)   # 3
print(ds.input_ids[0] )     # tensor([1, 2, 3, 4])
print(ds.target_ids[0])     # tensor([2, 3, 4, 5])
print(ds.input_ids[2] )     # tensor([5, 6, 7, 8])
print(ds.target_ids[2])     # tensor([6, 7, 8, 9])