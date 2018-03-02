import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from stability_selection import StabilitySelection

RNG_SEED = 123321


def test_stability_selection():

    p = 1000
    n = 1000
    k = 5
    rng = check_random_state(RNG_SEED)

    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    betas = np.zeros(p)
    chosen_betas = rng.choice(a=np.arange(p), size=k)
    betas[chosen_betas] = rng.uniform(size=k)

    probs = 1 / (1 + np.exp(-1 * np.matmul(X, betas)))
    y = (probs > 0.5).astype(int)

    selector = StabilitySelection()
    selector.fit(X, y)
    selector.get_support(indices=True)

