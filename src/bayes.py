import numpy as np


ln2pi = np.log(2 * np.pi)


def normal_log_likelihood(x, mu, sigma):
    return - ((x-mu)**2 / sigma + ln2pi + np.log(sigma)) / 2


def laplace(scale):
    U = np.random.uniform(-.5,.5, scale.shape)
    return -scale*np.sign(U)*np.log(1 - 2*np.abs(U))


def compute_per_class(n_classes, X, y, f):
    return np.array([f(X[y == c]) for c in range(n_classes)])


def positive(x):
    return np.maximum(x, 1e-6) if x is not None else None


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, data, y = None):
        pc, mu, sigma, cc = data
        self.lpp = np.log(pc) - np.log(pc.sum())
        self.mu = mu
        self.sigma = sigma
        self.lcp = np.log(cc) - np.log(pc)[:,None] if cc is not None else None
        return self

    def predict_log_proba(self, X):
        X_num, X_cat = X
        log_probs = self.lpp
        if self.mu is not None:
            log_probs = log_probs + normal_log_likelihood(X_num[:,None,:], self.mu, self.sigma).sum(-1)
        if self.lcp is not None:
            log_probs = log_probs + self.lcp[:,X_cat].sum(-1).T
        return log_probs

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis = -1)

    @property
    def params(self):
        pp = np.exp(self.lpp)[:,None]
        if self.mu is None:
            return pp, np.exp(self.lcp)
        if self.lcp is None:
            return pp, self.mu, self.sigma
        return pp, self.mu, self.sigma, np.exp(self.lcp)


class CentralizedStatistics:
    def __init__(self, eps, n_classes, n_num_feats, n_cat_vals):
        self.eps = eps
        self.n_classes = n_classes
        self.n_num_feats = n_num_feats
        self.n_cat_vals = n_cat_vals

    def fit_transform(self, data, y = None):
        (X_num, X_cat), y = data

        pc = np.unique(y, return_counts = True)[1].astype(float)

        if self.n_num_feats > 0:
            mu = compute_per_class(self.n_classes, X_num, y, lambda X: X.mean(0))
            sigma = compute_per_class(self.n_classes, X_num, y, lambda X: X.var(0))
        else:
            mu = None
            sigma = None

        if self.n_cat_vals > 0:
            cc = compute_per_class(self.n_classes, X_cat, y, lambda X: np.bincount(X.flatten(), minlength = self.n_cat_vals).astype(float))
        else:
            cc = None

        if self.eps is not None:
            pc += laplace(np.ones_like(pc) / self.eps)
            cc = cc + laplace(np.ones_like(cc) / self.eps) if cc is not None else None
            if mu is not None:
                d = compute_per_class(self.n_classes, X_num, y, lambda X: X.max(0) - X.min(0))
                n_samples = y.shape[0]
                sens = d / (n_samples + 1)
                mu += laplace(sens / self.eps)
                sigma += laplace(np.sqrt(n_samples) * sens / self.eps)

        return positive(pc), mu, positive(sigma), positive(cc)

    def transform(self, X):
        return X


class FederatedStatistics:
    def __init__(self, eps, n_classes, n_num_feats, n_cat_vals):
        self.eps = eps
        self.n_classes = n_classes
        self.n_num_feats = n_num_feats
        self.n_cat_vals = n_cat_vals

    def fit_transform(self, nodes, y = None):
        for node in nodes:
            node.eps = self.eps
            node.n_classes = self.n_classes
            node.n_cat_vals = self.n_cat_vals

        pc = sum(node.pc for node in nodes)

        if self.n_num_feats > 0:
            mu = sum(node.s for node in nodes) / pc[:,None]
            sigma = sum(node.q for node in nodes) / pc[:,None] - mu**2
        else:
            mu = None
            sigma = None

        if self.n_cat_vals > 0:
            cc = sum(node.cc for node in nodes)
        else:
            cc = None

        return pc, mu, positive(sigma), positive(cc)

    def transform(self, X):
        return X

class Node:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def count_dp(self, X, N):
        if X.shape[0] == 0:
            return np.zeros(N)
        count = np.bincount(X.flatten(), minlength = N).astype(float)
        if self.eps is not None:
            count += laplace(np.ones_like(count) / self.eps)
        return positive(count)

    def sum_dp(self, X):
        if X.shape[0] == 0:
            return np.zeros(X.shape[1])
        sum = X.sum(0)
        if self.eps is not None:
            sens = np.abs(X).max(0)
            sum += laplace(sens / self.eps)
        return sum

    @property
    def pc(self):
        return self.count_dp(self.y, self.n_classes)

    @property
    def s(self):
        return compute_per_class(self.n_classes, self.X[0], self.y, lambda X: self.sum_dp(X))

    @property
    def q(self):
        return compute_per_class(self.n_classes, self.X[0], self.y, lambda X: np.maximum(self.sum_dp(X**2), 0))

    @property
    def cc(self):
        return compute_per_class(self.n_classes, self.X[1], self.y, lambda X: self.count_dp(X, self.n_cat_vals))

    @property
    def s_sens(self):
        return compute_per_class(self.n_classes, self.X[0], self.y, lambda X: np.abs(X).max(0) if X.shape[0] > 0 else np.zeros(X.shape[1]))

    @property
    def q_sens(self):
        return compute_per_class(self.n_classes, self.X[0], self.y, lambda X: np.abs(X**2).max(0) if X.shape[0] > 0 else np.zeros(X.shape[1]))
