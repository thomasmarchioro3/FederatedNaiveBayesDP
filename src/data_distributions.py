import numpy as np

from .bayes import Node

class BaseDataDistributor:
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        return self.name if len(self.args) == 0 else f'{self.name}({",".join(map(str,self.args))})'

    def transform(self, X):
        return X

class IID(BaseDataDistributor):
    def __init__(self):
        super().__init__('IID')

    def fit_transform(self, data, y = None):
        num_points_per_node, ((X_num, X_cat), y) = data
        shuffle = np.random.permutation(y.shape[0])
        X_num = X_num[shuffle]
        X_cat = X_cat[shuffle]
        y = y[shuffle]
        offsets = np.zeros(len(num_points_per_node) + 1, dtype = int)
        np.cumsum(num_points_per_node, out = offsets[1:])
        nodes = [Node(X = (X_num[start:stop], X_cat[start:stop]), y = y[start:stop]) for start, stop in zip(offsets[:-1], offsets[1:])]
        return nodes