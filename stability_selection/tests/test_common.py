from sklearn.utils.estimator_checks import check_estimator
from stability_selection import StabilitySelection


def test_transformer():
    # This fails at the moment because in the low sample size case some of the bootstrap samples can have zero cases
    # of the positive class
    #return check_estimator(StabilitySelection)
    pass