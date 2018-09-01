import numpy as np
from numpy.testing import assert_almost_equal

from nose.tools import raises
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix

from stability_selection import StabilitySelection, RandomizedLasso, \
    RandomizedLogisticRegression


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


def test_estimator():
    check_estimator(RandomizedLasso)
    check_estimator(RandomizedLogisticRegression)


@raises(ValueError)
def test_logistic_weakness():
    n, p = 200, 200
    rho = 0.6

    X, y = generate_experiment_data(n, p, rho)
    RandomizedLogisticRegression(weakness=0.0).fit(X, y)


@raises(ValueError)
def test_logistic_weakness():
    n, p = 200, 200
    rho = 0.6

    X, y = generate_experiment_data(n, p, rho)
    RandomizedLasso(weakness=0.0).fit(X, y)


def test_randomized_lasso():
    n, p = 200, 200
    rho = 0.6
    weakness = 0.2

    X, y = generate_experiment_data(n, p, rho)
    lambda_grid = np.linspace(0.01, 0.5, num=100)

    estimator = RandomizedLasso(weakness=weakness)
    selector = StabilitySelection(base_estimator=estimator, lambda_name='alpha',
                                  lambda_grid=lambda_grid, threshold=0.9, verbose=1)
    selector.fit(X, y)

    chosen_betas = selector.get_support(indices=True)

    assert_almost_equal(np.array([0, 1]), chosen_betas)


def test_issparse():
    n, p = 200, 200
    rho = 0.6
    weakness = 0.2

    X, y = generate_experiment_data(n, p, rho)
    lambda_grid = np.linspace(0.01, 0.5, num=100)

    estimator = RandomizedLasso(weakness=weakness)
    selector = StabilitySelection(base_estimator=estimator, lambda_name='alpha',
                                  lambda_grid=lambda_grid, threshold=0.9, verbose=1)
    selector.fit(csr_matrix(X), y)
