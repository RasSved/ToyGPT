from layers import layer_norm, gelu, linear, feed_forward, add_residual
from attention import (attention_scores, softmax, context_vector, 
                       project, scale_scores, self_attention_all_queries, 
                       self_attention_single_query, causal_mask, causal_self_attention,
                       multi_head_attention)


# Pre norm transformer block where we combine attention and layer
def transformer_block(x_seq, heads, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2):
    normalized = layer_norm(x_seq, gamma1, beta1)
    m_h_attention = multi_head_attention(normalized, heads)
    result = add_residual(x_seq, m_h_attention)
    normalized2 = layer_norm(result, gamma2, beta2)
    ff = feed_forward(normalized2, W1, b1, W2, b2)
    result2 = add_residual(ff, result)
    return result2