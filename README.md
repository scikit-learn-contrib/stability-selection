# stability-selection - A scikit-learn compatible implementation of stability selection

[![Travis Status](https://travis-ci.org/thuijskens/stability-selection.svg?branch=master)](https://travis-ci.org/thuijskens/stability-selection)
[![Coverage Status](https://coveralls.io/repos/github/thuijskens/stability-selection/badge.svg?branch=master)](https://coveralls.io/github/thuijskens/stability-selection?branch=master)
[![CircleCI Status](https://circleci.com/gh/thuijskens/stability-selection.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/thuijskens/stability-selection/tree/master)

**stability-selection** is a Python implementation of the stability selection algorithm[^1], following
the scikit-learn `Estimator` API.

## Installation and usage

Before installing the module you will need `numpy`, `matplotlib`, and `sklearn`
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
selector = StabilitySelection(alphas=np.logspace(-5, -1, 50))
selector.fit(X, y)

print(selector.get_support(indices=True))
```

## Algorithmic details

Stability selection is a technique that aims to enhance and improve existing feature 
selection algorithms. For a generic feature selection algorithm, we have a tuning 
parameter $\lambda \in \Lambda$ that controls the amount of regularisation. Examples 
of such algorithms are:

1. $\ell_1$-penalized regression (penalization parameter $\lambda$).
2. Orthogonal matching pursuit (number of steps in forward selection).
3. Boosting ($\ell_1$ penalty)

These structure learning algorithms have in common is a parameter $\lambda \in \Lambda$
that controls the amount of regularisation. For every value of $\lambda$, we obtain a structure 
estimate $S^\lambda = \{1, \ldots, p\}$, which indicates which variables to select. We are
interested to determine whether there exists a $\lambda$ such that $S^\lambda$ is identical to 
$S$ with high probability, and how to achieve the right amount of regularisation.


Sability selection works as follows:

1. Define a candidate set of regularization parameters $\Lambda$ and a subsample number $N$. 
2. For each value $\lambda \in \Lambda$ do:

    a. For each $i$ in $\{1, \ldots, N\}$, do:
    
        i. Generate a bootstrap sample of the original data $X^{n \times p}$ of size $\frac{n}{2}$.
        ii. Run the selection algorithm (LASSO) on the bootstrap sample with regularization parameter $\lambda$.
    
    b. Given the selection sets from each subsample, calculate the empirical selection probability for each model component:
$$
\hat{\Pi}^\lambda_k = \mathbb{P}[k \in \hat{S}^\lambda] = \frac{1}{N} \sum_{i = 1}^N \mathbb{I}_{\{k \in \hat{S}_i^\lambda\}}.
$$

    c. The selection probability for component $k$ is its probability of being selected by the algorithm.

3. Given the selection probabilities for each component and for each value of $\lambda$, construct the 
   stable set according to the following definition:

$$
\hat{S}^{\text{stable}} = \{k : \max_{\lambda \in \Lambda} \hat{\Pi}_k^\lambda \geq \pi_\text{thr}\}
$$
   where $\pi_\text{thr}$ is a predefined threshold.

This algorithm identifies a set of “stable” variables that are selected with high probability.

## References

[^1]: Meinshausen, N. and Buhlmann, P., 2010. Stability selection. Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 72(4), pp.417-473.
