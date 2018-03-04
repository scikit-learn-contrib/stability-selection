import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.utils import check_random_state
from stability_selection import StabilitySelection, plot_stability_path


def _generate_dummy_classification_data(p=1000, n=1000, k=5, random_state=123321):

    rng = check_random_state(random_state)

    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    betas = np.zeros(p)
    important_betas = np.sort(rng.choice(a=np.arange(p), size=k))
    betas[important_betas] = rng.uniform(size=k)

    probs = 1 / (1 + np.exp(-1 * np.matmul(X, betas)))
    y = (probs > 0.5).astype(int)

    return X, y, important_betas


def test_stability_selection():
    n, k = 1000, 5

    X, y, important_betas = _generate_dummy_classification_data(n=n, k=k)
    selector = StabilitySelection(alphas=np.logspace(-5, -1, 25))
    selector.fit(X, y)

    chosen_betas = selector.get_support(indices=True)
    X_r = selector.transform(X)

    fig, ax = plot_stability_path(selector)

    assert_almost_equal(important_betas, chosen_betas)
    assert(X_r.shape == (n, k))
    assert(selector.stability_scores_.shape == (selector.n_bootstrap_iterations, selector.alphas.shape[0]))

