# Federated Naive Bayes under Differential Privacy

This code can be used to reproduce the results from the paper: "Federated Naive Bayes under Differential Privacy".

Scripts:
- `python src/run.py` to compute the accuracy of Naive Bayes classifier (standard, centralized DP, federated DP).
- `python src/measure_error.py` to compute the errors in the parameter estimation when using DP.
- `python src/measure_sensitivity.py` to compute the sensitivity of the queries.

Hyperparameters of the Monte Carlo simulations can be changed from `src/config.py`.
