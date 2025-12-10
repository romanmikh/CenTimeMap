# pylint: disable=no-member, arguments-differ, no-name-in-module

import torch
from torch import Tensor, tensor
from torchmetrics import Metric


class RAE(Metric):
    """
    Relative Absolute Error metric that handles both censored and uncensored data in survival analysis.
    """

    def __init__(self, mode: str = "uncensored"):
        """
        Initialize the RAE metric.
        RAE = sum_i^N frac{|y_i - hat{y_i}|}{y_i}

        Args:
            mode (str): The mode of the metric. Either "uncensored" or "censored".
        """
        super().__init__()
        self.mode = mode
        assert mode in [
            "uncensored",
            "censored",
        ], f"Invalid mode: {mode}. Must be 'uncensored' or 'censored'."

        self.add_state("rel_abs_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, event: Tensor) -> None:
        """
        Update the metric state.

        Args:
            preds (Tensor): The model predictions.
            target (Tensor): The target values.
            event (Tensor): The event indicator.
        """
        preds = preds.squeeze()
        target = target.squeeze()
        event = event.squeeze().bool()
        if self.mode == "uncensored":
            self.rel_abs_error += torch.sum(
                torch.abs(preds[event == 1] - target[event == 1]) / target[event == 1]
            )
            self.total += torch.sum(event == 1)
        else:
            self.rel_abs_error += torch.sum(
                torch.abs(preds[event == 0] - target[event == 0]) / target[event == 0]
            )
            self.total += torch.sum(event == 0)

    def compute(self) -> Tensor:
        """
        Compute the metric.

        Returns:
            Tensor: The computed RAE.
        """
        return self.rel_abs_error / self.total  # type: ignore
