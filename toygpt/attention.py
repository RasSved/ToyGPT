import math

# We want to return a score based on how similiar the input is to the query
def attention_scores(query, inputs):
    result = []
    for vector in inputs:
        score = 0
        for a, b in zip(vector, query):
            score += a * b
        result.append(score)
        
    return result

# Turning the attention scores in to actaul usefull softmax scores
# softmax(x) = exp(x) / sum(exp(x_i) for all x_i in the list
def softmax(scores):
    result = []
    total = sum(math.exp(score) for score in scores)
    for x in scores:
        smax = math.exp(x) / total
        result.append(smax)

    return result

# Scale each input vector by its attention weight, then sum them into one context vector
def context_vector(weights, inputs):
    res = []
    for weight, inp in zip(weights, inputs):
        weighted = [weight * i for i in inp]
        res.append(weighted)

    return [sum(x) for x in zip(*res)]

# Simply does the same as the context vecotr (matrix-vector multiplication)
def project(x, W):
    res = []
    for x_i, row in zip(x, W):
        x_weighted = [x_i * j for j in row]
        res.append(x_weighted)
    return [sum(x) for x in zip(*res)]

# Scale attention scores down by sqrt(d_k) before softmax, to prevent large dot products
# from producing extremely peaked (near one-hot) softmax outputs
def scale_scores(scores, d_k):
    scale = math.sqrt(d_k)
    return [score / scale for score in scores]


# Full scaled dot product sefl attention for one query token, 
# project Q/K/V, score query against keys, scale, softmax then blend into one context vector
def self_attention_single_query(inputs, W_query, W_key, W_value, query_idx):

    q = project(inputs[query_idx], W_query)

    keys = []
    values = []
    for vec in inputs:
        key = project(vec, W_key)
        value = project(vec, W_value)
        keys.append(key)
        values.append(value)

    a_q_score = attention_scores(q, keys)
    d_k = len(q)
    scaled = scale_scores(a_q_score, d_k)
    smax = softmax(scaled)
    context = context_vector(smax, values)
    return context

# Same as for one query token but we now made it for all, also moved out keys and values so they are not recalculated every time
def self_attention_all_queries(inputs, W_query, W_key, W_value):
    keys = [project(vec, W_key) for vec in inputs]
    values = [project(vec, W_value) for vec in inputs]

    result = []
    for query_idx in range(len(inputs)):
        q = project(inputs[query_idx], W_query)
        a_q_score = attention_scores(q, keys)
        d_k = len(q)
        scaled = scale_scores(a_q_score, d_k)
        smax = softmax(scaled)
        context = context_vector(smax, values)
        result.append(context)

    return result

# We want to take make sure that each position can only attend itself and earlier tokens so 
# when we generate we cant "peek" at later words
def causal_mask(scores, query_idx):
    if query_idx >= len(scores):
        raise ValueError("query idx is outside the list")

    result = scores.copy()
    result[query_idx + 1:] = [-float("inf")] * (len(scores) - (query_idx + 1))

    return result

# This is the same self attention all queries but we added the causal masking so we
# can use it for generating text
def causal_self_attention(inputs, W_query, W_key, W_value):
    keys = [project(vec, W_key) for vec in inputs]
    values = [project(vec, W_value) for vec in inputs]

    result = []
    for query_idx in range(len(inputs)):
        q = project(inputs[query_idx], W_query)
        a_q_score = attention_scores(q, keys)
        d_k = len(q)
        scaled = scale_scores(a_q_score, d_k)
        causal_masked = causal_mask(scaled, query_idx)
        smax = softmax(causal_masked)
        context = context_vector(smax, values)
        result.append(context)

    return result

# Run causal self attention over each head using the same input and finally combinding 
# into a list of vectors where each vector represents one token over the heads
def multi_head_attention(inputs, heads):
    one_list = []
    for head in heads:
        W_query, W_key, W_value = head
        causal_masked = causal_self_attention(inputs, W_query, W_key, W_value)
        one_list.append(causal_masked)
    result = [[x for lst in group for x in lst] for group in zip(*one_list)]
    return result

