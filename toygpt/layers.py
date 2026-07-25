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


def feed_forward(x, W1, b1, W2, b2):
    

x = [1.0, 2.0]
W = [[1.0, 0.0, 1.0],
     [0.0, 1.0, 1.0]]   # d_in=2 rows, d_out=3 cols
b = [0.5, 0.5, 0.5]

print(linear(x, W, b))
# x @ W = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1.0, 2.0, 3.0]
# + b   = [1.5, 2.5, 3.5]