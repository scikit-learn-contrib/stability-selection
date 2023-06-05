"""
===========================
Randomized LASSO example
===========================

An example plot of the stability scores for each variable after fitting :class:`stability_selection.StabilitySelection`
with :class:`stability_selection.RandomizedLasso`
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
    lambda_grid = np.linspace(0.001, 0.5, num=100)

    for weakness in [0.2, 0.5, 1.0]:
        estimator = RandomizedLasso(weakness=weakness,normalize=True)
        selector = StabilitySelection(base_estimator=estimator, lambda_name='alpha',lambda_grid=lambda_grid,
                                      threshold=0.9, verbose=1)
        selector.fit(X, y)

        fig, ax = plot_stability_path(selector)
        fig.show()






