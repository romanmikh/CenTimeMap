"""
A class specific to cox models as we need to accumulate predictions, targets, and events
during training and use them to infer the median survival time using the Breslow
estimator.
"""

# pylint: disable=no-member, arguments-differ, attribute-defined-outside-init
# mypy: ignore-errors
# pylance: reportOptionalMemberAccess=false

from typing import Optional

import numpy as np
import torch
from sksurv.linear_model.coxph import BreslowEstimator
from torch import Tensor
from torchmetrics import Metric


class CoxAccumulator(Metric):
    """
    Accumulator for Cox models.
    """

    def __init__(self, tmax: int) -> None:
        super().__init__()
        self.add_state("tr_preds", default=Tensor([]), dist_reduce_fx="cat")
        self.add_state("tr_events", default=Tensor([]), dist_reduce_fx="cat")
        self.add_state("tr_times", default=Tensor([]), dist_reduce_fx="cat")
        self.add_state("ts_preds", default=Tensor([]), dist_reduce_fx="cat")
        self.tmax = tmax

    def update(
        self,
        tr_preds: Optional[Tensor] = None,
        tr_events: Optional[Tensor] = None,
        tr_times: Optional[Tensor] = None,
        ts_preds: Optional[Tensor] = None,
    ) -> None:
        """
        Update the accumulator with predictions and targets. During training, we
        accumulate the training set predictions, events, and times. During testing, we
        accumulate the test set predictions.

        Args:
            tr_preds (Tensor): Training set predicted risk scores. tr_events (Tensor):
            Training set binary event labels. tr_times (Tensor): Training set
            time-to-event labels. ts_preds (Tensor): Test set predicted risk scores.
        """
        if tr_preds is not None:
            # keep tensors 1-D
            tr_preds  = tr_preds.flatten()
            tr_events = tr_events.flatten()          # type: ignore
            tr_times  = tr_times.flatten()           # type: ignore
            self.tr_preds = torch.cat([self.tr_preds, tr_preds])
            self.tr_events = torch.cat([self.tr_events, tr_events])
            self.tr_times = torch.cat([self.tr_times, tr_times])
        else:
            assert (
                ts_preds is not None
            ), "Either training stats or test predictions must be provided."
            ts_preds = ts_preds.flatten()  # type: ignore
            self.ts_preds = torch.cat([self.ts_preds, ts_preds])

    def fit_breslow(self):
        """
        Fit the Breslow estimator on the training set.
        """
        tr_preds = self.tr_preds.cpu().numpy()
        tr_events = self.tr_events.cpu().numpy()
        tr_times = self.tr_times.cpu().numpy()
        self.breslow = BreslowEstimator().fit(tr_preds, tr_events, tr_times)

        min_time, max_time = tr_times.min(), tr_times.max()
        self.times = np.arange(min_time, max_time)

    def compute(self, training: bool = False) -> Tensor:
        """
        Compute the median survival time from the predicted risk scores using Cox model.

        Args:
            training (bool, optional): Whether to compute the median survival time for
            the training set. Defaults to False.

        Returns:
            torch.Tensor: Predicted median survival time for each sample in the test set.
        """
        if training:
            sample_surv_fn = self.breslow.get_survival_function(
                self.tr_preds.cpu().numpy()
            )
        else:
            sample_surv_fn = self.breslow.get_survival_function(
                self.ts_preds.cpu().numpy()
            )

        intermediate_preds = np.asarray(
            [[fn(t) for t in self.times] for fn in sample_surv_fn]
        )

        preds = []
        for pred in intermediate_preds:
            median_time_idx = np.where(pred <= 0.5)[0]
            preds.append(median_time_idx[0] if len(median_time_idx) > 0 else self.tmax)

        return torch.Tensor(preds)
