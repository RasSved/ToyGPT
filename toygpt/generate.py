from model import gpt_model, GPTModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# Greedy choice pick of taking whichever vocab entry scored the highest
def next_token_id(logits):
    best = logits[0]
    best_ind = 0
    for i, v in enumerate(logits):
        if logits[i] > best:
            best = v
            best_ind = i

    return best_ind


# Reshape before softmax without changing which token "won"
def apply_temperature(logits, temperature):
    return logits / temperature

# We want to filter out bad tokens before we even start to look at probabilities
def apply_top_k(logits, k):
    if k < len(logits):
        top = torch.topk(logits, k)
        result = torch.full_like(logits, fill_value=-float("inf"))
        result[top.indices] = top.values
        return result
    else:
        return logits

# Combined step where we replace greedy pick prediction,
# Apply topk if gven -> scale down with temp -> softmax -> draw index by prob 
def sample_next_token(logits, temperature=1.0, top_k=None):
    if top_k != None:
        logits = apply_top_k(logits, top_k)
    tempe = apply_temperature(logits, temperature)
    softm = F.softmax(tempe, dim=-1)
    prob = torch.multinomial(softm, num_samples=1)
    return int(prob)

# KV-cache for real models TODO
# Call gpt model to get logits then with our greedy pick we get the prediction for "next word"
# This is repeated for however many new token we want
def generatev1(ids, num_new_tokens, token_embedding_matrix, pos_embedding_matrix, block_params_list, gamma_f, beta_f, W_out, b_out):
    for _ in range(num_new_tokens):
        gpt = gpt_model(ids, token_embedding_matrix, pos_embedding_matrix, block_params_list, gamma_f, beta_f, W_out, b_out)
        prediction = next_token_id(gpt[-1])
        ids = ids.copy()
        ids.append(prediction)
    return ids


# Cropp the input ids if they are outside tokenlen range -> get logits from our model -> 
# look at the last logits and get a next token prediction from it -> add the prediction at the end 
# of our ids. (Basically adding a word at the end of a sentence using our prediction)
def generate(model, ids, num_new_tokens, context_length, temperature=1.0, top_k=None): 
    for _ in range(num_new_tokens):
        cropped = ids if ids.shape[1] <= context_length else ids[:, -context_length:]
        logits = model(cropped)
        gpt = logits[0, -1, :]
        next_t = sample_next_token(gpt, temperature, top_k)
        ids = torch.cat([ids, torch.tensor([[next_t]])], dim=1)
    return ids 

torch.manual_seed(123)

model = GPTModel(vocab_size=10, d_model=8, num_heads=2, num_layers=2, d_ff=16, context_length=6)
ids = torch.tensor([[1, 2, 3]])   # shape (1, 3) -- note the batch dimension

torch.manual_seed(0)
out1 = generate(model, ids, num_new_tokens=4, context_length=6, temperature=1.0, top_k=None)
print(out1)   # tensor([[1, 2, 3, 6, 1, 2, 3]])

torch.manual_seed(0)
out2 = generate(model, ids, num_new_tokens=4, context_length=6, temperature=1.0, top_k=3)
print(out2)   # tensor([[1, 2, 3, 4, 1, 0, 2]])

torch.manual_seed(0)
out3 = generate(model, ids, num_new_tokens=8, context_length=6, temperature=1.0, top_k=None)
print(out3)          # tensor([[1, 2, 3, 6, 1, 2, 3, 1, 8, 1, 4]])
print(out3.shape[1]) # 11
