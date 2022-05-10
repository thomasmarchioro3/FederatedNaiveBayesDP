import multiprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
import sys
import time
from tqdm import tqdm

from .bayes import NaiveBayes, CentralizedStatistics, FederatedStatistics
from . import config
from .data_distributions import IID
from . import data_loading
from .node_sizes import EqualSizes


def run(cfg):
    dataset, eps, iters, *fed_args = cfg
    (X_train, y_train, X_test, y_test), n_classes, n_num_feats, n_cat_feats, n_cat_vals = getattr(data_loading, f'load_{dataset}')()
    scaled_eps = None if eps is None else eps / (1 + 2*n_num_feats + n_cat_feats)

    if len(fed_args) == 0:
        conf_str = f'\n{dataset},{eps},centralized,,,'
        pipeline = Pipeline([
            ('statistics_computer', CentralizedStatistics(scaled_eps, n_classes, n_num_feats, n_cat_vals)),
            ('naive_bayes', NaiveBayes())
        ])
    else:
        N, node_size_computer, data_distributor = fed_args
        if 2*N > y_train.shape[0]:
            return ''
        conf_str = f'\n{dataset},{eps},federated,{N},{node_size_computer},{data_distributor}'
        pipeline = Pipeline([
            ('node_size_computer', node_size_computer(N)),
            ('data_distributor', data_distributor),
            ('statistics_computer', FederatedStatistics(scaled_eps, n_classes, n_num_feats, n_cat_vals)),
            ('naive_bayes', NaiveBayes())
        ])

    results = ''
    for _ in range(iters):
        pipeline.fit((X_train, y_train))
        y_pred = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        train_bal_acc = balanced_accuracy_score(y_train, y_pred)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        results += f'{conf_str},{train_acc},{train_bal_acc},{test_acc},{test_bal_acc}'

    return results


def generate_configs():
    for dataset in config.datasets:
        for eps in config.eps_levels:
            yield (dataset, eps, config.iters)
            for N in config.num_nodes:
                if N == 1:
                    yield (dataset, eps, config.iters, 1, EqualSizes(), IID())
                else:
                    for node_size_computer in config.node_size_computers:
                        for data_distributor in config.data_distributors:
                            yield (dataset, eps, config.iters, N, node_size_computer, data_distributor)


def main_multiprocess(f, configs, workers):
    with multiprocessing.Pool(processes = workers) as pool, \
         tqdm(total = len(configs)) as progress:
        for results in pool.imap_unordered(run, configs):
            f.write(results)
            progress.update()


def main_singleprocess(f, configs):
    for cfg in tqdm(configs):
        f.write(run(cfg))


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) >= 2 else None

    all_configs = list(generate_configs())

    t = time.strftime('%Y%m%dT%H%M%S')
    with open(f'results_{t}.csv', 'w') as f:
        f.write('dataset,eps,kind,nodes,node_size_distrib,data_distrib,train_acc,train_bal_acc,test_acc,test_bal_acc')

        if workers is not None:
            main_multiprocess(f, all_configs, workers)
        else:
            main_singleprocess(f, all_configs)