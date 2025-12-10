"""
CoxMB Trainer.
"""

import os
import torch
import wandb
from torch import nn
from typing import Tuple
from src.methods.cox_trainer import CoxTrainer
from src.methods.mb import CoxMemoryBank
from src.utils.train_utils import get_grad_norm


class CoxMBTrainer(CoxTrainer):
    """
    Trainer for CoxMB loss.
    """

    def __init__(self, k: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        total_samples = len(self.train_loader.dataset)  # type: ignore
        self.memory_bank = CoxMemoryBank(
            k=k, total_samples=total_samples, device=self.device
        )
        self.exp_name = "CoxMB" + self.exp_name

        if wandb.run is not None:
            wandb.finish()

        wandb.init(project="centime", name=self.exp_name)
        os.makedirs(f"runs/{self.exp_name}")
        self.yaml.update({"k": k, "loss_fn": "cox_loss", "exp_name": self.exp_name})
        self.save_args()

    def train_one_epoch(self) -> Tuple[float, float, float, float]:
        """
        Train the model for one epoch using the Cox + Memory Bank method.
        """
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            img = batch["img"]
            clinical_data = batch["clinical_data"] if self.clinical_data else None
            self.optimizer.zero_grad()
            output = self.model(img, clinical_data)
            if output.dim() > 1:                                    # collapse histogram to expected month
                t_idx  = torch.arange(1, output.size(-1) + 1,       
                                      device=output.device).float()
                output = (output * t_idx).sum(dim=-1, keepdim=True) # risk = -E[T] TODO: double check the sign
            # update memory bank
            self.memory_bank.update(output, batch["event"], batch["time"])

            loss = self.compute_loss(*self.memory_bank.get_memory_bank())
            loss.backward()
            # free gradients
            self.memory_bank.free_gradients()

            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(  # type: ignore
                    self.model.parameters(), self.clip_grad_norm
                )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            grad_norm = get_grad_norm(self.model)

            train_loss += (
                loss.item() * self.memory_bank.get_memory_bank()[2].sum().item()
            )

            wandb.log(
                {
                    "train_loss_batch": loss.item(),
                    "grad_norm": grad_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            self.cindex.update(output, batch["time"], batch["event"])
            self.accumulator.update(
                tr_preds=output, tr_events=batch["event"], tr_times=batch["time"]
            )

        train_loss /= self.train_loader.dataset.events.sum().item()  # type: ignore
        cindex, mae_nc, mae_c = self.get_metrics(training=True)

        # reset memory bank
        self.memory_bank.reset()

        return train_loss, cindex, mae_nc, mae_c
