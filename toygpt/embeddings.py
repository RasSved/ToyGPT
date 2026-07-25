
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

# Element vise addition between vektors
def combine_embeddings(token_embeddings, pos_embeddings):
    
    if len(token_embeddings) != len(pos_embeddings):
        raise ValueError("Vektors are different length")

    result = list()
    for lst, lst2 in zip(token_embeddings, pos_embeddings):
        if len(lst) != len(lst2):
            raise ValueError("Lists are different lengths")
        combined = list()
        for i in range(len(lst)):
            combined.append(lst[i] + lst2[i])
        result.append(combined)
        

    return result
