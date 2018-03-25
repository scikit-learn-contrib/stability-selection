"""
===============================
Bootstrap helper functions
===============================

This module contains helper functions for stability_selection.py that do bootstrap sampling
"""

import numpy as np

from sklearn.utils.random import sample_without_replacement


def bootstrap_without_replacement(n_samples, n_subsamples, random_state=None):
    """
    Bootstrap without replacement. It is a wrapper around
    sklearn.utils.random.sample_without_replacement.

    Parameters
    ----------
    n_samples : int
        Number of total samples
    n_subsamples : int
        Number of subsamples in the bootstrap sample
    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    out : array of size [n_subsamples,]
            The sampled subsets of integer. The subset of selected integer might
            not be randomized, see the method argument.
    """
    return sample_without_replacement(n_samples, n_subsamples, random_state=random_state)


def complementary_pairs_bootstrap(n_samples, n_subsamples, random_state=None):
    """
    Complementary pairs bootstrap. Two subsamples A and B are generated, such
    that |A| = n_subsamples, the union of A and B equals {0, ..., n_samples - 1},
    and the intersection of A and B is the empty set.

    Parameters
    ----------
    n_samples : int
        Number of total samples
    n_subsamples : int
        Number of subsamples in the bootstrap sample
    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    A : array of size [n_subsamples,]
            The sampled subsets of integer. The subset of selected integer might
            not be randomized, see the method argument.
    B : array of size [n_samples - n_subsamples,]
            The complement of A.
    """
    subsample = bootstrap_without_replacement(n_samples, n_subsamples, random_state)
    complementary_subsample = np.setdiff1d(np.arange(n_samples), subsample)

    return subsample, complementary_subsample