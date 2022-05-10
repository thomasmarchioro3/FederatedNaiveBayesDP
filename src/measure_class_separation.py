import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

from .bayes import CentralizedStatistics, NaiveBayes
from . import config
from . import data_loading


for dataset in config.datasets:
    (X_train, y_train, _, _), n_classes, n_num_feats, _, n_cat_vals = getattr(data_loading, f'load_{dataset}')()

    if n_num_feats > 0:
        pipeline = Pipeline([
            ('statistics_computer', CentralizedStatistics(None, n_classes, n_num_feats, n_cat_vals)),
            ('naive_bayes', NaiveBayes())
        ])
        pipeline.fit((X_train, y_train))
        _, mu, sigma, *_ = pipeline.named_steps['naive_bayes'].params

        Nc = mu.shape[0]
        feats = np.arange(mu.shape[1])
        width = 0.7 / Nc

        plt.figure()
        plt.title(dataset)
        for c, (m, s) in enumerate(zip(mu, sigma)):
            offset = width * (c + 1/2 - Nc/2)
            plt.errorbar(feats + offset, m, s, marker = 'o', ls = 'none', label = f'class {c}')
        plt.xlabel('feature')
        plt.xticks(feats, feats)
        plt.legend()

plt.show()