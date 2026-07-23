
# Look up token ids row in the matrix (id to vector)
def embedding_lookup(embedding_matrix, ids):
    result = list()
    for token_id in ids:
        result.append(embedding_matrix[token_id])
    return result

# Make sure we get the positions of words since that matters for context
def positional_embedding_lookup(pos_embedding_matrix, seq_len):
    if len(pos_embedding_matrix) < seq_len - 1:
        raise ValueError("seq_len out of bounds")
    pos = list(range(seq_len))
    
    return embedding_lookup(pos_embedding_matrix, pos)

def combine_embeddings(token_embeddings, pos_embeddings):

token_embeddings = [[0.1, 0.2], [0.3, 0.4]]
pos_embeddings   = [[0.9, 0.1], [0.8, 0.2]]

test = combine_embeddings(token_embeddings, pos_embeddings)
print(test)