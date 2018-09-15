# stability-selection - A scikit-learn compatible implementation of stability selection

[![Build Status](https://travis-ci.org/scikit-learn-contrib/stability-selection.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/stability-selection)
[![Coverage Status](https://coveralls.io/repos/github/scikit-learn-contrib/stability-selection/badge.svg?branch=master)](https://coveralls.io/github/scikit-learn-contrib/stability-selection?branch=master)
[![CircleCI](https://circleci.com/gh/scikit-learn-contrib/stability-selection.svg?style=svg)](https://circleci.com/gh/scikit-learn-contrib/stability-selection)

**stability-selection** is a Python implementation of the stability selection feature selection algorithm, first proposed by [Meinshausen and Buhlmann](https://stat.ethz.ch/~nicolai/stability.pdf). 

The idea behind stability selection is to inject more noise into the original problem by generating bootstrap samples of the data, and to use a base feature selection algorithm (like the LASSO) to find out which features are important in every sampled version of the data. The results on each bootstrap sample are then aggregated to compute a *stability score* for each feature in the data. Features can then be selected by choosing an appropriate threshold for the stability scores.

## Installation

To install the module, clone the repository
```bash
git clone https://github.com/scikit-learn-contrib/stability-selection.git
```
Before installing the module you will need `numpy`, `matplotlib`, and `sklearn`. Install these modules separately, or install using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
and execute the following in the project directory to install `stability-selection`:
```bash
python setup.py install
```

## Documentation and algorithmic details

See the [documentation](https://thuijskens.github.io/stability-selection/docs/index.html) for details on the module, and the accompanying [blog post](https://thuijskens.github.io/2018/07/25/stability-selection/) for details on the algorithmic details.

## Example usage

`stability-selection` implements a class `StabilitySelection`, that takes any scikit-learn compatible estimator that has either a ``feature_importances_`` or ``coef_`` attribute after fitting. Important other parameters are

- `lambda_name`: the name of the penalization parameter of the base estimator (for example, `C` in the case of `LogisticRegression`).
- `lambda_grid`: an array of values of the penalization parameter to iterate over.

After instantiation, the algorithm can be run with the familiar `fit` and `transform` calls.

### Basic example
See below for an example:
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

## This is all preparation of the dummy data set
n, p, k = 500, 1000, 5

X, y, important_betas = _generate_dummy_classification_data(n=n, k=k)
base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1'))
])

## Here stability selection is instantiated and run
selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__C',
                              lambda_grid=np.logspace(-5, -1, 50)).fit(X, y)

print(selector.get_support(indices=True))
```

### Bootstrapping strategies

`stability-selection` uses bootstrapping without replacement by default (as proposed in the original paper), but does support different bootstrapping strategies. [Shah and Samworth] proposed *complentairy pairs* bootstrapping, where the data set is bootstrapped in pairs, such that the intersection is empty but the union equals the original data set. `StabilitySelection` supports this through the `bootstrap_func` parameter. 

This parameter can be:
- A string, which must be one of
    - 'subsample': For subsampling without replacement (default).
    - 'complementary_pairs': For complementary pairs subsampling [2].
    - 'stratified': For stratified bootstrapping in imbalanced
       classification.
- A function that takes `y`, and a random state
  as inputs and returns a list of sample indices in the range
  `(0, len(y)-1)`. 

For example, the `StabilitySelection` call in the above example can be replaced with 
```python
selector = StabilitySelection(base_estimator=base_estimator,
                              lambda_name='model__C',
                              lambda_grid=np.logspace(-5, -1, 50),
                              bootstrap_func='complementary_pairs')
selector.fit(X, y)
```
to run stability selection with complementary pairs bootstrapping.

## Feedback and contributing

Feedback and contributions are much appreciated. If you have any feedback, please post it on the [issue tracker](https://github.com/scikit-learn-contrib/stability-selection/issues). 

## References

[1]: Meinshausen, N. and Buhlmann, P., 2010. Stability selection. Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 72(4), pp.417-473.
    
[2] Shah, R.D. and Samworth, R.J., 2013. Variable selection with
   error control: another look at stability selection. Journal
   of the Royal Statistical Society: Series B (Statistical Methodology),
    75(1), pp.55-80.
