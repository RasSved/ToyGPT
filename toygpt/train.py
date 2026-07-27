import torch
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, input_batch, target_batch, optimizer):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = F.cross_entropy(output, target_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

