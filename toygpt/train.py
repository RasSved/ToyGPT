import torch
import torch.nn as nn
import torch.nn.functional as F

# Trainging step: Clear gradiant from prev -> run model to get logits -> find loss from those digits compared to target ->
# call backward to compute gradiant -> call optimizer step to actaully update weigths
def train_step(model, input_batch, target_batch, optimizer):
    optimizer.zero_grad()
    loss = calc_loss_batch(model, input_batch, target_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


# For each epoch we feed the given batch from dataloader into training step then gather the losses
def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    output = []
    for epoch in range(num_epochs):
        for input_batch, target_batch in dataloader:
            loss = train_step(model, input_batch, target_batch, optimizer)
            output.append(loss)
    return output

# Same as trainstep but without the training we are only instrested in the actual loss
def calc_loss_batch(model, input_batch, target_batch):
    output = model(input_batch)
    vocab_size = output.shape[-1]
    loss = F.cross_entropy(output.view(-1, vocab_size), target_batch.view(-1))
    return loss

# Calls our loss batch calc and from those losses we take the avg returning a final avg loss int 
def calc_loss_loader(model, dataloader, num_batches=None):
    loss = []
    with torch.no_grad():   
        for num, (input_idx, target_idx) in enumerate(dataloader):
            if num_batches != None and num == num_batches:
                break
            else:

                loss.append(calc_loss_batch(model, input_idx, target_idx).item())
        avg = sum(loss) / len(loss)
        return avg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import GPTDataset

# small deterministic corpus, vocab_size=8
token_ids = [1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2]
ds = GPTDataset(token_ids, context_size=4, stride=2)
dl = DataLoader(ds, batch_size=2, shuffle=False)

class TinyModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.out = nn.Linear(d_model, vocab_size)
    def forward(self, ids):
        return self.out(self.emb(ids))

torch.manual_seed(42)
model = TinyModel(vocab_size=8, d_model=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

print(calc_loss_loader(model, dl))                  # num_batches=None -> average all 3
# 2.4091386795043945   ((2.401893+2.423917+2.401606)/3)

print(calc_loss_loader(model, dl, num_batches=2))   # average first 2 only
# 2.412905216217041     ((2.401893+2.423917)/2)

print(calc_loss_loader(model, dl, num_batches=1))   # just the first batch
# 2.401893138885498     -- should exactly equal batch 0's individual loss