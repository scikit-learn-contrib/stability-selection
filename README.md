# stability-selection - A scikit-learn compatible implementation of stability selection

[![Travis Status](https://travis-ci.org/thuijskens/stability-selection.svg?branch=master)](https://travis-ci.org/thuijskens/stability-selection)
[![Coverage Status](https://coveralls.io/repos/github/thuijskens/stability-selection/badge.svg?branch=master)](https://coveralls.io/github/thuijskens/stability-selection?branch=master)
[![CircleCI Status](https://circleci.com/gh/thuijskens/stability-selection.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/thuijskens/stability-selection/tree/master)

**stability-selection** is a Python implementation of the stability selection algorithm[^1], following
the scikit-learn `Estimator` API.

## Installation and usage

Before installing the module you will need `numpy`, `matplotlib`, and `sklearn`.
To install the module, clone the repository
```shell
git clone https://github.com/thuijskens/stability-selection.git
``` 
and execute the following in the project directory:
```shell
python setup.py install
```

## Example usage

```python
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from stability_selection import StabilitySelection


def _generate_dummy_classification_data(p=1000, n=1000, k=5, random_state=123321):

    rng = check_random_state(random_state)

    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    betas = np.zeros(p)
    important_betas = np.sort(rng.choice(a=np.arange(p), size=k))
    betas[important_betas] = rng.uniform(size=k)

    probs = 1 / (1 + np.exp(-1 * np.matmul(X, betas)))
    y = (probs > 0.5).astype(int)

    return X, y, important_betas


n, p, k = 500, 1000, 5

X, y, important_betas = _generate_dummy_classification_data(n=n, k=k)
base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1'))
])
selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__C',
                              lambda_grid=np.logspace(-5, -1, 50)).fit(X, y)

print(selector.get_support(indices=True))
```

## Algorithmic details

See the [documentation](https://thuijskens.github.io/stability-selection/docs/index.html)

## References

[^1]: Meinshausen, N. and Buhlmann, P., 2010. Stability selection. Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 72(4), pp.417-473.
