from model import gpt_model

# Greedy choice pick of taking whichever vocab entry scored the highest
def next_token_id(logits):
    best = logits[0]
    best_ind = 0
    for i, v in enumerate(logits):
        if logits[i] > best:
            best = v
            best_ind = i

    return best_ind

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

