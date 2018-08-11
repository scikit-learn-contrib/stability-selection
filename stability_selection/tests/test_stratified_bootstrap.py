import numpy as np
from nose.tools import raises

from stability_selection.bootstrap import stratified_bootstrap


@raises(ValueError)
def test_check_not_classification():
    y = np.linspace(0, 1, 21)
    stratified_bootstrap(y, 10, random_state=0)


def test_stratified_bootstrap():
    zero_to_one_ratio = 3
    n_ones = 10

    y = np.array(n_ones * ([0] * zero_to_one_ratio + [1]))
    for n_subsamples in [4, 8, 12, 16, 20]:
        sample_idx = stratified_bootstrap(y, n_subsamples, random_state=0)
        samples = y[sample_idx]

        assert(len(samples) == n_subsamples)

        n_ones = (samples == 1).sum()
        n_zeros = (samples == 0).sum()
        assert(n_zeros == n_ones * zero_to_one_ratio)


def test_random_state():
    zero_to_one_ratio = 3
    n_ones = 10

    y = np.array(n_ones * ([0] * zero_to_one_ratio + [1]))

    samples0 = np.sort(stratified_bootstrap(y, 12, random_state=0))
    samples0b = np.sort(stratified_bootstrap(y, 12, random_state=0))
    samples1 = np.sort(stratified_bootstrap(y, 12, random_state=1))

    assert(np.array_equal(samples0, samples0b))
    assert(not np.array_equal(samples0, samples1))
