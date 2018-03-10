"""
===========================
Randomized LASSO example
===========================

An example plot of the stability scores for each variable after fitting :class:`stability_selection.stability_selection.StabilitySelection`
"""

import numpy as np

from sklearn.utils import check_random_state
from stability_selection import StabilitySelection, RandomizedLasso, plot_stability_path


def generate_experiment_data(n=200, p=200, rho=0.6, random_state=3245):
    rng = check_random_state(random_state)

    sigma = np.eye(p)
    sigma[0, 2] = rho
    sigma[2, 0] = rho
    sigma[1, 2] = rho
    sigma[2, 1] = rho

    X = rng.multivariate_normal(mean=np.zeros(p), cov=sigma, size=(n,))
    beta = np.zeros(p)
    beta[:2] = 1.0
    epsilon = rng.normal(0.0, 0.25, size=(n,))

    y = np.matmul(X, beta) + epsilon

    return X, y


if __name__ == '__main__':
    n, p = 200, 200
    rho = 0.6

    X, y = generate_experiment_data()

    for weakness in [0.2, 0.5, 1.0]:
        estimator = RandomizedLasso(normalize=True)
        selector = StabilitySelection(base_estimator=estimator, alphas=np.logspace(-5, -1, 50))
        selector.fit(X, y)

        fig, ax = plot_stability_path(selector)
        fig.show()






