import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import GPTDataset

# Trainging step: Clear gradiant from prev -> run model to get logits -> find loss from those digits compared to target ->
# call backward to compute gradiant -> call optimizer step to actaully update weigths
def train_step(model, input_batch, target_batch, optimizer):
    optimizer.zero_grad()
    loss = calc_loss_batch(model, input_batch, target_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


# For each epoch we feed the given batch from train loader into training step 
# for each eval_freq step we evaluate the model and return train and val loss as 2 lists
def train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_num_batches):
    model.train()
    train_losses = []
    val_losses = []
    global_step_counter = 0
    for epoch in range(num_epochs):
        for input_idx, target_idx in train_loader:
            train_step(model, input_idx, target_idx, optimizer)
            global_step_counter += 1
            if global_step_counter % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_num_batches)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
    return train_losses, val_losses

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

# Get loss of traning and val loader
def evaluate_model(model, train_loader, val_loader, num_batches=None):
    model.eval()
    train_loss = calc_loss_loader(model, train_loader, num_batches)
    val_loss = calc_loss_loader(model, val_loader, num_batches)
    model.train()
    return (train_loss, val_loss)

# Simply saving a chekpoint by taking the state of the model and optimzer
# and saving them together with what epoch we are on to path
def save_checkpoint(model, optimizer, epoch, path):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    torch.save({"model_state_dict": model_state, 
                "optimizer_state_dict": optimizer_state, 
                "epoch": epoch}, path)


# Load the stored wiegths to the model and optimzer not creating a new one
# aswell as return the epoch it was on when saved
def load_checkpoint(model, optimzer, path):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimzer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch
