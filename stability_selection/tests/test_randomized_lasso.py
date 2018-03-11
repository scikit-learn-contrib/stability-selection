import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.utils import check_random_state
from stability_selection import StabilitySelection, RandomizedLasso


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


def test_randomized_lasso():
    n, p = 200, 200
    rho = 0.7
    weakness = 0.2

    X, y = generate_experiment_data(n, p, rho)
    lambda_grid = np.linspace(0.001, 0.5, num=100)

    estimator = RandomizedLasso(weakness=weakness)
    selector = StabilitySelection(base_estimator=estimator, lambda_name='alpha',
                                  lambda_grid=lambda_grid, threshold=0.9, verbose=1)
    selector.fit(X, y)

    chosen_betas = selector.get_support(indices=True)

    assert_almost_equal(np.array([0, 1]), chosen_betas)