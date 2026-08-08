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

# Data prep util where we turn ids into maxlength by either cutting or padding 
# with a given token 
def pad_or_truncate(ids, max_length, pad_token_id):
    if len(ids) >= max_length:
        return ids[:max_length]
    
    copy = ids.copy()
    while len(copy) < max_length:
        copy.append(pad_token_id)
    return copy

# Here we want to find what our max length actaully is and is used by looking at the 
# longest seq in the dataset
def longest_sequence_length(encoded_text):
    if encoded_text:
        return len(max(encoded_text, key=len))
    else:
        raise ValueError("Empty Text Encoding")

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

# Data set for complaints where instead of sliding windowd get texts with labels, we encode the texts 
# and pass the output in order to get encoding + labels pairs
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, encode_fn, pad_token_id, max_length=None):
        self.texts = texts
        self.labels = torch.LongTensor(labels)
        self.encode_fn = encode_fn
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.reshaped = []
        
        encoded = [self.encode_fn(text) for text in self.texts]
        if self.max_length is None:
            self.max_length = longest_sequence_length(encoded)

        for encoded_text in encoded:
            padded = pad_or_truncate(encoded_text, self.max_length, self.pad_token_id)
            self.reshaped.append(torch.LongTensor(padded))
        
    def __len__(self):
        return len(self.reshaped)
    
    def __getitem__(self, idx):
        return (self.reshaped[idx], self.labels[idx])




