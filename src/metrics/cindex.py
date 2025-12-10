""" Concordance Index metric for survival analysis. """

# pylint: disable=no-member, attribute-defined-outside-init, arguments-differ
from typing import Union

import torch
from sksurv.metrics import concordance_index_censored
from torch import Tensor
from torchmetrics import Metric


class CIndex(Metric):
    """
    Implements the concordance index metric for survival analysis.
    """

    def __init__(self) -> None:
        """
        Initialize the metric.
        """
        super().__init__()
        self.add_state("preds", default=Tensor([]), dist_reduce_fx="cat")
        self.add_state("targets", default=Tensor([]), dist_reduce_fx="cat")
        self.add_state("events", default=Tensor([]).bool(), dist_reduce_fx="cat")

    def update(self, preds: Tensor, targets: Tensor, events: Tensor) -> None:
        """Shapes are forced to (N,) even when N == 1."""
        preds   = preds.view(-1)
        targets = targets.view(-1)
        events  = events.view(-1).bool()

        self.preds   = torch.cat((self.preds, preds))    
        self.targets = torch.cat((self.targets, targets))
        self.events  = torch.cat((self.events, events))  


    def compute(self) -> Tensor:
        """
        Compute the concordance index.

        Returns:
            Tensor: The concordance index.
        """
        res = concordance_index_censored(
            self.events.cpu().numpy(),
            self.targets.cpu().numpy(),
            self.preds.cpu().numpy(),
        )[0]
        return torch.Tensor([res])
