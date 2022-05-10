import heapq
import numpy as np
from typing import Callable, List

def compute_node_sizes(n_points: int, n_nodes: int, node_weight: Callable[[int],float]) -> List[int]:
    assert n_points >= 2*n_nodes, f'Cannot split {n_points} data points across {n_nodes} nodes'

    relative_weights = np.array([node_weight(n) for n in range(n_nodes)])
    min_pop = relative_weights.min()
    if (min_pop) < 0:
        relative_weights += min_pop

    divisor = relative_weights.sum() / n_nodes
    num_points_per_node = (relative_weights / divisor).round().astype(int)
    num_points_per_node[num_points_per_node < 2] = 2

    tot_points = num_points_per_node.sum()
    if tot_points > n_points:
        queue = [(rem, i) for i, rem in enumerate(relative_weights - num_points_per_node * divisor)]
        heapq.heapify(queue)
        while tot_points > n_points:
            _, i = queue[0]
            if num_points_per_node[i] > 2:
                num_points_per_node[i] -= 1
                heapq.heapreplace(queue, (relative_weights[i] - num_points_per_node[i] * divisor, i))
                tot_points -= 1
    elif tot_points < n_points:
        queue = [(rem, i) for i, rem in enumerate(num_points_per_node * divisor - relative_weights)]
        heapq.heapify(queue)
        while tot_points < n_points:
            _, i = queue[0]
            num_points_per_node[i] += 1
            heapq.heapreplace(queue, (num_points_per_node[i] * divisor - relative_weights[i], i))
            tot_points +=1

    assert num_points_per_node.sum() == n_points, 'Something went wrong'
    assert (num_points_per_node >= 2).all(), 'Something went very wrong'

    return num_points_per_node


class BaseSizeComputer:
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __call__(self, n_nodes):
        self.n_nodes = n_nodes
        return self

    def __repr__(self) -> str:
        return self.name if len(self.args) == 0 else f'{self.name}({",".join(map(str,self.args))})'

    def fit_transform(self, data, y = None):
        n_points = data[1].shape[0]
        num_points_per_node = compute_node_sizes(n_points, self.n_nodes, self.weight_func(self.n_nodes))
        return (num_points_per_node, data)

    def transform(self, X):
        return X


class EqualSizes(BaseSizeComputer):
    def __init__(self):
        super().__init__('Equal')

    def weight_func(self, N):
        return lambda _: 1


class UniformSizes(BaseSizeComputer):
    def __init__(self, low, high):
        super().__init__('Uniform', low, high)

    def weight_func(self, N):
        return np.random.uniform(*self.args, N).__getitem__


class ExponentialSizes(BaseSizeComputer):
    def __init__(self, scale):
        super().__init__('Exponential', scale)

    def weight_func(self, N):
        return np.random.exponential(*self.args, N).__getitem__