# pylint: disable=no-member
import torch

from src.metrics import MAE


def test_mae():
    """
    Test the MAE metric.
    """
    # censored case where all events are censored
    mae = MAE(mode="censored")
    preds = torch.Tensor([0.0, 1.0, 2.0])
    target = torch.Tensor([1.0, 2.0, 2.0])
    event = torch.Tensor([0, 0, 0])
    mae.update(preds, target, event)
    assert mae.compute() == 2 / 3.0

    # uncensored case where all events are uncensored
    mae = MAE(mode="uncensored")
    event = torch.Tensor([1, 1, 1])
    mae.update(preds, target, event)
    assert mae.compute() == 2 / 3.0

    # mixed case where some events are censored
    mae = MAE(mode="uncensored")
    event = torch.Tensor([1, 0, 1])
    mae.update(preds, target, event)
    assert mae.compute() == 0.5

    mae = MAE(mode="censored")
    mae.update(preds, target, event)
    assert mae.compute() == 1

if __name__ == "__main__":
    test_mae()
    print("MAE tests passed.")
