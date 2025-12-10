# pylint: disable=no-member
import torch
from torch.testing import assert_close

from src.metrics import RAE


def test_rae():
    """
    Test the RAE metric.
    """
    # censored case where all events are censored
    rae = RAE(mode="censored")
    preds = torch.Tensor([0.0, 1.0, 2.0])
    target = torch.Tensor([1.0, 2.0, 2.0])
    event = torch.Tensor([0, 0, 0])
    rae.update(preds, target, event)
    assert rae.compute() == 0.5

    # uncensored case where all events are uncensored
    rae = RAE(mode="uncensored")
    event = torch.Tensor([1, 1, 1])
    rae.update(preds, target, event)
    assert rae.compute() == 0.5

    # mixed case where some events are censored
    rae = RAE(mode="uncensored")
    event = torch.Tensor([1, 0, 1])
    rae.update(preds, target, event)
    assert rae.compute() == 0.5

    rae = RAE(mode="censored")
    rae.update(preds, target, event)
    assert rae.compute() == 0.5

    # test reset
    rae = RAE(mode="uncensored")
    rae.update(preds, target, event)
    rae.reset()
    assert rae.rel_abs_error == 0.0
    assert rae.total == 0.0

    # test multiple updates
    rae = RAE(mode="uncensored")
    rae.update(preds, target, event)
    rae.update(preds, target, event)
    assert rae.compute() == 0.5

    # another mixed case
    rae = RAE(mode="uncensored")
    preds = torch.Tensor([0.0, 1.0, 2.0, 3.0])
    target = torch.Tensor([1.0, 2.0, 2.0, 5.0])
    event = torch.Tensor([1, 0, 1, 1])
    rae.update(preds, target, event)
    assert_close(rae.compute().item(), 0.46666667)

if __name__ == "__main__":
    test_rae()
    print("RAE tests passed.")