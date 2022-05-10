import multiprocessing
import numpy as np
import os
from sklearn.pipeline import Pipeline
import sys
import time
from tqdm import tqdm

from .bayes import CentralizedStatistics, FederatedStatistics
from . import config
from .data_distributions import IID
from . import data_loading
from .node_sizes import EqualSizes


def run(cfg):
    base_folder, dataset, eps, iters, *fed_args = cfg
    (X_train, y_train, _, _), n_classes, n_num_feats, n_cat_feats, n_cat_vals = getattr(data_loading, f'load_{dataset}')()
    scaled_eps = None if eps is None else eps / (1 + 2*n_num_feats + n_cat_feats)

    perfect_pipeline = Pipeline([
        ('statistics_computer', CentralizedStatistics(None, n_classes, n_num_feats, n_cat_vals))
    ])
    perfect_params = np.concatenate(filter(lambda x: x is not None, perfect_pipeline.fit_transform((X_train, y_train)), axis = 1))

    if len(fed_args) == 0:
        pipeline = Pipeline([
            ('statistics_computer', CentralizedStatistics(scaled_eps, n_classes, n_num_feats, n_cat_vals))
        ])
        save_location = f'{base_folder}/{dataset}_{round(eps, 9)}_centr.npy'
    else:
        N, node_size_computer, data_distributor = fed_args
        if 2*N > y_train.shape[0]:
            return
        pipeline = Pipeline([
            ('node_size_computer', node_size_computer(N)),
            ('data_distributor', data_distributor),
            ('statistics_computer', FederatedStatistics(scaled_eps, n_classes, n_num_feats, n_cat_vals))
        ])
        save_location = f'{base_folder}/{dataset}_{round(eps, 9)}_fed_{N}_{node_size_computer}_{data_distributor}.npy'

    errors = np.empty((iters, n_classes, (1 + 2*n_num_feats + n_cat_vals)))
    for i in range(iters):
        noisy_params = np.concatenate(filter(lambda x: x is not None, pipeline.fit_transform((X_train, y_train)), axis = 1))
        errors[i] = (noisy_params - perfect_params) / perfect_params

    np.save(save_location, errors) # iters x n_classes x (1 + 2*n_num_feats + n_cat_vals)


def generate_configs(base_folder):
    for dataset in ['adult']: #config.datasets:
        for eps in config.eps_levels:
            if eps is None:
                continue
            # yield (base_folder, dataset, eps, config.iters)
            for N in config.num_nodes:
                if N == 1:
                    yield (base_folder, dataset, eps, config.iters, 1, EqualSizes(), IID())
                else:
                    for node_size_computer in config.node_size_computers:
                        for data_distributor in config.data_distributors:
                            yield (base_folder, dataset, eps, config.iters, N, node_size_computer, data_distributor)


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
    base_folder = f'errors_{t}'
    os.makedirs(base_folder)

    all_configs = list(generate_configs(base_folder))

    if workers is not None:
        main_multiprocess(all_configs, workers)
    else:
        main_singleprocess(all_configs)
