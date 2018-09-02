import numpy as np

from nose.tools import raises
from numpy.testing import assert_array_equal
from sklearn.utils.estimator_checks import check_estimator
from stability_selection import StabilitySelection


def test_transformer():
    # With defaults this can fail because in the low sample size case
    # some of the bootstrap samples can have zero cases of the positive class
    return check_estimator(StabilitySelection(n_bootstrap_iterations=10, sample_fraction=1.0))


@raises(ValueError)
def test_check_string_threshold():
    StabilitySelection(threshold='wrong_value')._validate_input()


@raises(ValueError)
def test_check_threshold_too_large():
    StabilitySelection(threshold=1.5)._validate_input()


@raises(ValueError)
def test_check_threshold_too_small():
    StabilitySelection(threshold=0.0)._validate_input()


@raises(ValueError)
def test_check_threshold_too_small():
    StabilitySelection().get_support(threshold='wrong_value')


@raises(ValueError)
def test_check_arguments():
    StabilitySelection(threshold='wrong_value')._validate_input()


@raises(ValueError)
def test_check_wrong_lambda_name():
    StabilitySelection(lambda_name='alpha')._validate_input()


@raises(ValueError)
def test_check_wrong_lambda_name():
    StabilitySelection(n_bootstrap_iterations=-1)._validate_input()


def test_automatic_lambda_grid():
    selector = StabilitySelection()
    selector._validate_input()
    assert_array_equal(np.logspace(-5, -2, 25), selector.lambda_grid)


@raises(ValueError)
def test_bootstrap_func():
    StabilitySelection(bootstrap_func='nonexistent')._validate_input()


@raises(ValueError)
def test_callable_bootstrap_func():
    StabilitySelection(bootstrap_func=0.5)._validate_input()


@raises(ValueError)
def test_sample_fraction():
    StabilitySelection(sample_fraction=0.0)._validate_input()


@raises(ValueError)
def test_lambda_name():
    StabilitySelection(lambda_name='n_estimators')._validate_input()
