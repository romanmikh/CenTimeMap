# pylint: disable=arguments-differ

from torch import Tensor
from torchmetrics.regression import MeanAbsoluteError


class MAE(MeanAbsoluteError):
    """
    Mean Absolute Error metric wrapper to handle computing MAE for both censored and
    uncensored data in survival analysis.
    """

    def __init__(self, mode: str = "uncensored", **kwargs):
        """
        Initialize the MAE metric.

        Args:
            mode (str): The mode of the metric. Either "uncensored" or "censored".
        """
        super().__init__(**kwargs)
        self.mode = mode
        assert mode in [
            "uncensored",
            "censored",
        ], f"Invalid mode: {mode}. Must be 'uncensored' or 'censored'."

    def update(self, preds: Tensor, target: Tensor, event: Tensor) -> None:  # type: ignore
        """
        Update the metric state.

        Args:
            preds (Tensor): The model predictions.
            target (Tensor): The target values.
            event (Tensor): The event indicator.
        """
        preds  = preds.view(-1)
        target = target.view(-1)
        event  = event.view(-1).bool()

        if self.mode == "uncensored":
            super().update(preds[event == 1], target[event == 1])
        elif self.mode == "censored":
            super().update(preds[event == 0], target[event == 0])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
