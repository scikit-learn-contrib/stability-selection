import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import raises

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from stability_selection import StabilitySelection, plot_stability_path


def _generate_dummy_regression_data(p=1000, n=1000, k=5,
                                    random_state=123321):
    rng = check_random_state(random_state)

    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    betas = np.zeros(p)
    important_betas = np.sort(rng.choice(a=np.arange(p), size=k))
    betas[important_betas] = rng.uniform(size=k)

    y = np.matmul(X, betas)

    return X, y, important_betas


def _generate_dummy_classification_data(p=1000, n=1000, k=5,
                                        random_state=123321):

    rng = check_random_state(random_state)

    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    betas = np.zeros(p)
    important_betas = np.sort(rng.choice(a=np.arange(p), size=k))
    betas[important_betas] = rng.uniform(size=k)

    probs = 1 / (1 + np.exp(-1 * np.matmul(X, betas)))
    y = (probs > 0.5).astype(int)

    return X, y, important_betas


def test_stability_selection_classification():
    n, p, k = 1000, 1000, 5

    X, y, important_betas = _generate_dummy_classification_data(n=n, k=k)
    selector = StabilitySelection(lambda_grid=np.logspace(-5, -1, 25), verbose=1)
    selector.fit(X, y)

    chosen_betas = selector.get_support(indices=True)
    X_r = selector.transform(X)

    assert_almost_equal(important_betas, chosen_betas)
    assert(X_r.shape == (n, k))
    assert(selector.stability_scores_.shape == (p, selector.lambda_grid.shape[0]))


def test_stability_selection_regression():
    n, p, k = 500, 1000, 5

    X, y, important_betas = _generate_dummy_regression_data(n=n, k=k)

    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    lambdas_grid = np.logspace(-1, 1, num=10)

    selector = StabilitySelection(base_estimator=base_estimator,
                                  lambda_name='model__alpha',
                                  lambda_grid=lambdas_grid)
    selector.fit(X, y)

    chosen_betas = selector.get_support(indices=True)

    assert_almost_equal(important_betas, chosen_betas)


def test_with_complementary_pairs_bootstrap():
    n, p, k = 500, 1000, 5

    X, y, important_betas = _generate_dummy_regression_data(n=n, k=k)

    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    lambdas_grid = np.logspace(-1, 1, num=10)

    selector = StabilitySelection(base_estimator=base_estimator,
                                  lambda_name='model__alpha',
                                  lambda_grid=lambdas_grid,
                                  bootstrap_func='complementary_pairs')
    selector.fit(X, y)

    chosen_betas = selector.get_support(indices=True)

    assert_almost_equal(important_betas, chosen_betas)


def test_with_stratified_bootstrap():
    n, p, k = 1000, 1000, 5

    X, y, important_betas = _generate_dummy_classification_data(n=n, k=k)
    selector = StabilitySelection(lambda_grid=np.logspace(-5, -1, 25), verbose=1,
                                  bootstrap_func='stratified')
    selector.fit(X, y)

    chosen_betas = selector.get_support(indices=True)
    assert_almost_equal(important_betas, chosen_betas)


@raises(ValueError)
def test_different_shape():
    n, p, k = 100, 200, 5

    X, y, important_betas = _generate_dummy_regression_data(n=n, k=k)

    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    lambdas_grid = np.logspace(-1, 1, num=10)

    selector = StabilitySelection(base_estimator=base_estimator,
                                  lambda_name='model__alpha',
                                  lambda_grid=lambdas_grid)
    selector.fit(X, y)
    selector.transform(X[:, :-2])


def test_no_features():
    n, p, k = 100, 200, 0

    X, y, important_betas = _generate_dummy_regression_data(n=n, k=k)

    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    lambdas_grid = np.logspace(-1, 1, num=10)

    selector = StabilitySelection(base_estimator=base_estimator,
                                  lambda_name='model__alpha',
                                  lambda_grid=lambdas_grid)
    selector.fit(X, y)

    assert_almost_equal(selector.transform(X),
                        np.empty(0).reshape((X.shape[0], 0)))


def test_stability_plot():
    n, p, k = 500, 200, 5

    X, y, important_betas = _generate_dummy_regression_data(n=n, k=k)

    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    lambdas_grid = np.logspace(-1, 1, num=10)

    selector = StabilitySelection(base_estimator=base_estimator,
                                  lambda_name='model__alpha',
                                  lambda_grid=lambdas_grid)
    selector.fit(X, y)

    plot_stability_path(selector, threshold_highlight=0.5)
