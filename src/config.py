from .data_distributions import IID
from .node_sizes import EqualSizes, UniformSizes, ExponentialSizes
import numpy as np

datasets = ['accelerometer', 'adult', 'votes', 'mushroom', 'digits', 'skin', 'heart']

eps_levels = (10 ** np.arange(-2, 1 + 1e-6, 0.5)).tolist() + [None]

num_nodes = [1, 10, 100, 1000]

node_size_computers = [
    EqualSizes(),
    UniformSizes(1, 10),
    ExponentialSizes(2),
]

data_distributors = [
    IID(),
]

iters = 1000