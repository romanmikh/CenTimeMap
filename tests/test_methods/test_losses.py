"""
Unit tests for the losses module.
"""

# pylint: disable=no-member
import torch
from numpy.testing import assert_allclose

from src.methods.losses import (
    centime_loss,
    classical_loss,
    get_mean_prediction,
    get_probs,
    get_survival_function,
)

torch.manual_seed(0)
torch.random.manual_seed(0)
LOC = torch.randint(0, 100, (4, 1))
SCALE = torch.ones_like(LOC) * 144
TMAX = 100
DELTA = torch.tensor([1, 1, 0, 0]).view(-1, 1)
TIME = torch.randint(0, TMAX, (4, 1))
PROB = get_probs(LOC, SCALE, TMAX)
MEAN_PREDS = torch.tensor([44.0066, 39.0278, 33.1224, 59.984])
SURV_DISTS = torch.nn.functional.softmax(torch.randn(4, TMAX) * 0.02 + 0.1, dim=1)
CLS_LOSS = (0.086381, 2.297787)
CENTIME_LOSS = (1.9071, 2.2978)


def test_get_survival_function():
    """
    Test the survival function.
    """
    surv = get_survival_function(PROB)
    assert surv.shape == (4, TMAX)
    for i in range(4):
        assert_allclose(surv[i, 0], 1.0, rtol=1e-4)


def test_get_mean_prediction():
    """
    Test the mean prediction.
    """
    mean_pred = get_mean_prediction(PROB, TMAX)
    for i, pred in enumerate(mean_pred):
        assert_allclose(pred, MEAN_PREDS[i], rtol=1e-4)


def test_classical_loss():
    """
    Test the classical loss.
    """
    cens_loss, uncens_loss = classical_loss(SURV_DISTS, DELTA, TIME, TMAX)
    assert_allclose(cens_loss, CLS_LOSS[0], rtol=1e-4)
    assert_allclose(uncens_loss, CLS_LOSS[1], rtol=1e-4)


def test_centime_loss():
    """
    Test the centime loss.
    """
    cens_loss, uncens_loss = centime_loss(SURV_DISTS, DELTA, TIME, TMAX)
    assert_allclose(cens_loss, CENTIME_LOSS[0], rtol=1e-4)
    assert_allclose(uncens_loss, CENTIME_LOSS[1], rtol=1e-4)


if __name__ == "__main__":
    test_get_survival_function()
    test_get_mean_prediction()
    test_classical_loss()
    test_centime_loss()
    print("All tests passed, check for assertion errors.")