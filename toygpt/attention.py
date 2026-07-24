import math

# We want to return a score based on how similiar the input is to the query
def attention_scores(query, inputs):
    result = list()
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
    result = list()
    total = sum(math.exp(score) for score in scores)
    for x in scores:
        smax = math.exp(x) / total
        result.append(smax)
    return result

def context_vector(wigths, inputs):
    pass

weights = [0.5, 0.2, 0.3]
inputs = [
    [1.0, 0.0],
    [0.0, 1.0],
    [2.0, 2.0],
]
test = context_vector(weights, inputs)

print(test)