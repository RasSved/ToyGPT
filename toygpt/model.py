from layers import layer_norm, linear, feed_forward, add_residual
from attention import multi_head_attention
from embeddings import embedding_lookup, positional_embedding_lookup, combine_embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# --------------------------------- USING PYTORCH ---------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_query = nn.Linear(d_model, d_model, bias=True)
        self.W_key   = nn.Linear(d_model, d_model, bias=True)
        self.W_value = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, seq_len, d_model = x.shape
        q = self.W_query(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(b, seq_len, d_model)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(approximate="tanh"), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, context_length, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, context_length, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))    # pre-norm: norm before sublayer, raw x in residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, context_length, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        b, seq_len = ids.shape
        positions = torch.arange(seq_len, device=ids.device)
        x = self.dropout(self.token_emb(ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        return self.out_head(self.final_norm(x))