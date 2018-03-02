from sklearn.utils.estimator_checks import check_estimator
from stability_selection import StabilitySelection


def test_transformer():
    return check_estimator(StabilitySelection)