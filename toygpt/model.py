from layers import layer_norm, linear, feed_forward, add_residual
from attention import multi_head_attention
from embeddings import embedding_lookup, positional_embedding_lookup, combine_embeddings


# Pre norm transformer block where we combine attention and layer
def transformer_block(x_seq, heads, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2):
    normalized = layer_norm(x_seq, gamma1, beta1)
    m_h_attention = multi_head_attention(normalized, heads)
    result = add_residual(x_seq, m_h_attention)
    normalized2 = layer_norm(result, gamma2, beta2)
    ff = feed_forward(normalized2, W1, b1, W2, b2)
    result2 = add_residual(ff, result)
    return result2

# Stacked the transformer block so we run the transformation params over x_seq N-time
def stack_transformer_blocks(x_seq, block_param):
    for block in block_param:
        x_seq = transformer_block(x_seq,*block)

    return x_seq   

# The final step before predictions where we take the outputed hidden vectors and apply
# on last normalization then project them from d_model to vocab_size sow e get a logit 
def output_head(x_seq, gamma_f, beta_f, W_out, b_out):
    normalized = layer_norm(x_seq, gamma_f, beta_f)
    output = []
    for vec in normalized:
        linearized = linear(vec, W_out, b_out)
        output.append(linearized)

    return output


# The function where we string everything together
# token embedding -> positional embedding -> combine embeddings -> 
# that combined over stacked attention blocks -> finnaly to logits via output head
def gpt_model(ids, token_embedding_matrix, pos_embedding_matrix, block_params_list, gamma_f, beta_f, W_out, b_out):
    t_embedding = embedding_lookup(token_embedding_matrix, ids)
    pos_embedding = positional_embedding_lookup(pos_embedding_matrix, len(ids))
    combined = combine_embeddings(t_embedding, pos_embedding)
    stacked = stack_transformer_blocks(combined, block_params_list)
    output = output_head(stacked, gamma_f, beta_f, W_out, b_out)
    return output

