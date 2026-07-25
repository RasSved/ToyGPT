import math

# Normalize the input vector to have zero mean and unit variance using the population 
# standard deviation with epsilon for numerical stability.
# Also added scaling with gamma and shifting with beta
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = sum(x) / len(x)
    centered = [i - mean for i in x]
    result = [i * i for i in centered]
    population_variance = sum(result) / len(result)

    std = math.sqrt(population_variance + eps)
    normalized = [(i / std) for i in centered]

    return [(a * b) + c for a,b,c in zip(normalized,gamma,beta)]

# Gaussian Error Linear Unit activation function
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * (x**3))))

# Matrix-Vector multiplication (x * W) + b where b is a bias vector
def linear(x, W, b):
    res = []
    for x_i, row in zip(x, W):
        x_weighted = [x_i * j for j in row]
        res.append(x_weighted)
    vec_mult = [sum(x) for x in zip(*res)]
    return [i + j for i, j in zip(vec_mult,b)]

# take input x and project it up with linear then use activation function on it and 
# scale it back down with linear
def feed_forward(x, W1, b1, W2, b2):
    l1 = linear(x, W1, b1)
    l1_g = [gelu(i) for i in l1]
    l2 = linear(l1_g, W2, b2)
    return l2

# Element vise addition between vektors (same as embeddings/combine_embeddings())
def add_residual(x, sublayer_output):
         
    if len(x) != len(sublayer_output):
        raise ValueError("Vektors are different length")

    result = []
    for lst, lst2 in zip(x, sublayer_output):
        if len(lst) != len(lst2):
            raise ValueError("Lists are different lengths")
        combined = []
        for i in range(len(lst)):
            combined.append(lst[i] + lst2[i])
        result.append(combined)

    return result

x_seq = [[1.0, 2.0], [0.5, 0.5]]
sublayer_output_seq = [[0.1, -0.2], [1.0, 1.0]]

print(add_residual(x_seq, sublayer_output_seq))
# [[1.1, 1.8], [1.5, 1.5]]

