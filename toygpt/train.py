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

losses = train_model(model, dl, optimizer, num_epochs=6)
print(len(losses))   # 6  (6 windows / batch_size=2 = 3 batches per epoch, x 2 epochs)
for l in losses:
    print(l)