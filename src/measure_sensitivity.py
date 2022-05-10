import multiprocessing
import numpy as np
import os
from sklearn.pipeline import Pipeline
import sys
import time
from tqdm import tqdm

from . import config
from .bayes import compute_per_class
from .data_distributions import IID
from . import data_loading
from .node_sizes import EqualSizes


def centralized_sensitivity(data, n_classes):
    (X_num, _), y = data
    d = compute_per_class(n_classes, X_num, y, lambda X: X.max(0) - X.min(0))
    n_samples = y.shape[0]
    mu_sens = d / (n_samples + 1)
    sigma_sens = np.sqrt(n_samples) * mu_sens
    return np.stack([mu_sens, sigma_sens])


class FederatedSensitivity:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self):
        raise NotImplementedError

    def compute_stats(self, x):
        q25, med, q75 = np.quantile(x, q = [.25, .5, .75], axis = 0)
        return np.stack([x.mean(0), x.min(0), x.max(0), x.var(0), q25, med, q75])

    def fit_transform(self, nodes, y = None):
        for node in nodes:
            node.n_classes = self.n_classes
        s_sens_stats = self.compute_stats(np.stack([node.s_sens for node in nodes]))
        q_sens_stats = self.compute_stats(np.stack([node.q_sens for node in nodes]))
        return np.stack((s_sens_stats, q_sens_stats))


def run(cfg):
    base_folder, dataset, *fed_args = cfg
    (X_train, y_train, _, _), n_classes, n_num_feats, _, _ = getattr(data_loading, f'load_{dataset}')()
    data = (X_train, y_train)

    if n_num_feats == 0:
        return

    if len(fed_args) == 0:
        results = centralized_sensitivity(data, n_classes)
        np.save(f'{base_folder}/{dataset}_centr.npy', results) # 2(mu/sigma) x n_classes x n_num_feats

    else:
        iters, N, node_size_computer, data_distributor = fed_args
        if 2*N > y_train.shape[0]:
            return
        pipeline = Pipeline([
            ('node_size_computer', node_size_computer(N)),
            ('data_distributor', data_distributor),
            ('sensitivity_computer', FederatedSensitivity(n_classes))
        ])
        results = np.stack([pipeline.fit_transform(data) for _ in range(iters)])
        np.save(f'{base_folder}/{dataset}_fed_{N}_{node_size_computer}_{data_distributor}.npy', results) # iters x 2(s/q) x 7(avg/min/max/var/q25/med/q75) x n_classes x n_num_feats


def generate_configs(base_folder):
    for dataset in config.datasets:
        yield (base_folder, dataset)
        for N in config.num_nodes:
            if N == 1:
                yield (base_folder, dataset, config.iters, 1, EqualSizes(), IID())
            else:
                for node_size_computer in config.node_size_computers:
                    for data_distributor in config.data_distributors:
                        yield (base_folder, dataset, config.iters, N, node_size_computer, data_distributor)


def main_multiprocess(configs, workers):
    with multiprocessing.Pool(processes = workers) as pool, \
         tqdm(total = len(configs)) as progress:
        for _ in pool.imap_unordered(run, configs):
            progress.update()


def main_singleprocess(configs):
    for cfg in tqdm(configs):
        run(cfg)


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) >= 2 else None

    t = time.strftime('%Y%m%dT%H%M%S')
    base_folder = f'sensitivities_{t}'
    os.makedirs(base_folder)
    
    all_configs = list(generate_configs(base_folder))

    if workers is not None:
        main_multiprocess(all_configs, workers)
    else:
        main_singleprocess(all_configs)