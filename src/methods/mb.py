"""
Memory Bank Class.
"""

# pylint: disable=no-member

from typing import Tuple, Union

import torch
from torch import Tensor


class CoxMemoryBank:
    """Memory Bank for CoxMB method."""

    def __init__(
        self,
        k: float = 1.0,
        total_samples: int = 0,
        device: Union[str, torch.device] = "cpu",
    ):
        self.max_len = int(total_samples * k)
        self.device = device

        self.preds = torch.empty(self.max_len, device=device)
        self.times = torch.empty(self.max_len, device=device, dtype=torch.int64)
        self.events = torch.empty(self.max_len, device=device, dtype=torch.int64)

        self.current_size = 0

    def update(self, preds: Tensor, events: Tensor, times: Tensor) -> None:
        batch_size = preds.size(0)

        preds = preds.squeeze()
        events = events.squeeze()
        times = times.squeeze()

        if self.current_size + batch_size <= self.max_len:
            self.preds[self.current_size : self.current_size + batch_size] = preds
            self.times[self.current_size : self.current_size + batch_size] = times
            self.events[self.current_size : self.current_size + batch_size] = events
        else:
            overflow = self.current_size + batch_size - self.max_len
            self.preds[:-overflow] = self.preds[overflow:].clone()
            self.times[:-overflow] = self.times[overflow:].clone()
            self.events[:-overflow] = self.events[overflow:].clone()

            if batch_size > self.max_len:
                # we need to use the last `max_len` elements of the batch
                self.preds[:] = preds[-self.max_len :]
                self.times[:] = times[-self.max_len :]
                self.events[:] = events[-self.max_len :]
            else:
                self.preds[-batch_size:] = preds
                self.times[-batch_size:] = times
                self.events[-batch_size:] = events

        self.current_size = min(self.current_size + batch_size, self.max_len)

    def get_memory_bank(self) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.preds[: self.current_size].unsqueeze(1),
            self.times[: self.current_size].unsqueeze(1),
            self.events[: self.current_size].unsqueeze(1),
        )

    def reset(self) -> None:
        self.preds = torch.empty(self.max_len, device=self.device)
        self.times = torch.empty(self.max_len, device=self.device, dtype=torch.int64)
        self.events = torch.empty(self.max_len, device=self.device, dtype=torch.int64)
        self.current_size = 0

    def free_gradients(self) -> None:
        """
        Free gradients of memory bank to avoid error in backpropagation.
        """
        self.preds = self.preds.detach()
