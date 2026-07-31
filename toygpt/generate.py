from model import gpt_model
import torch

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
    seen = {}
    



logits = torch.tensor([2.0, 1.0, 0.1, 5.0, -3.0])

print(apply_top_k(logits, k=2))
# tensor([2., -inf, -inf, 5., -inf])   -- only the top 2 (5.0 and 2.0) survive

print(apply_top_k(logits, k=3))
# tensor([2., 1., -inf, 5., -inf])     -- top 3 (5.0, 2.0, 1.0) survive

print(apply_top_k(logits, k=5))
# tensor([2.0, 1.0, 0.1, 5.0, -3.0])   -- k == vocab_size -> no-op, nothing filtered

# KV-cache for real models TODO
# Call gpt model to get logits then with our greedy pick we get the prediction for "next word"
# This is repeated for however many new token we want
def generate(ids, num_new_tokens, token_embedding_matrix, pos_embedding_matrix, block_params_list, gamma_f, beta_f, W_out, b_out):
    for t in range(num_new_tokens):
        gpt = gpt_model(ids, token_embedding_matrix, pos_embedding_matrix, block_params_list, gamma_f, beta_f, W_out, b_out)
        prediction = next_token_id(gpt[-1])
        ids = ids.copy()
        ids.append(prediction)
    return ids

