import math

# We want to return a score based on how similiar the input is to the query
def attention_scores(query, inputs):
    result = []
    for vector in inputs:
        score = 0
        for a, b in zip(vector, query):
            score += a * b
        i += 1
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
       
def project(x, W):

    d_in = len(W)
    d_out = len(W[0])
    pass


x = [1.0, 2.0]
W = [
    [1.0, 0.0, 2.0],   # row for x[0]
    [0.0, 1.0, 1.0],   # row for x[1]
]
test = project(x, W)
# output[0] = 1.0*1.0 + 2.0*0.0 = 1.0
# output[1] = 1.0*0.0 + 2.0*1.0 = 2.0
# output[2] = 1.0*2.0 + 2.0*1.0 = 4.0
# [1.0, 2.0, 4.0]

print(test)