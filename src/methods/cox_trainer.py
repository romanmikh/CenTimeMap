"""
Cox Trainer.
"""

# pylint: disable=arguments-differ

import os
from typing import Tuple, no_type_check

import torch
from torch import Tensor, nn
from torch.utils.data import SequentialSampler

import wandb
from src.methods.base_trainer import BaseTrainer
from src.methods.losses import cox_loss
from src.metrics.cox_accumulator import CoxAccumulator
from src.utils.train_utils import get_grad_norm


class CoxTrainer(BaseTrainer):
    """
    Trainer for Cox loss.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = cox_loss
        self.accumulator = CoxAccumulator(tmax=self.tmax).to(self.device)

        self.exp_name = "Cox" + self.exp_name

        if wandb.run is not None:
            wandb.finish()

        wandb.init(project="centime", name=self.exp_name)
        os.makedirs(f"runs/{self.exp_name}")
        self.yaml.update({"loss_fn": "cox_loss", "exp_name": self.exp_name})
        self.save_args()

    @no_type_check
    def compute_loss(self, output: Tensor, target: Tensor, event: Tensor) -> Tensor:
        """
        Compute the Cox loss.
        """
        return self.loss_fn(output, event, target)

    def get_predicted_time(self, output: Tensor) -> Tensor:
        """
        Get the predicted time of death from the model output.
        """
        raise NotImplementedError

    def train_one_epoch(
        self,
    ) -> Tuple[float, float, float, float]:
        """
        Train the model for one epoch using the Cox loss.
        """
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            img = batch["img"]
            clinical_data = batch["clinical_data"] if self.clinical_data else None
            self.optimizer.zero_grad()
            output = self.model(img, clinical_data)                 # (B, 1) or (B, tmax) if heads
            if output.dim() > 1:                                    # collapse histogram to expected month
                t_idx  = torch.arange(1, output.size(-1) + 1,       
                                      device=output.device).float()
                output = -(output * t_idx).sum(dim=-1, keepdim=True) # risk = -E[T]
            loss = self.compute_loss(output, batch["time"], batch["event"])
            loss.backward()
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(  # type: ignore
                    self.model.parameters(), self.clip_grad_norm
                )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            grad_norm = get_grad_norm(self.model)

            train_loss += loss.item() * batch["event"].sum().item()

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

        return train_loss, cindex, mae_nc, mae_c

    def validate_one_epoch(self) -> Tuple[float, float, float, float]:
        """
        Validate the model for one epoch using the Cox loss.
        """
        self.model.eval()
        val_loss = 0.0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            img = batch["img"]
            clinical_data = batch["clinical_data"] if self.clinical_data else None
            output = self.model(img, clinical_data)                 # (B, 1) or (B, tmax) if heads
            if output.dim() > 1:                                    # collapse histogram to expected month
                t_idx  = torch.arange(1, output.size(-1) + 1,       
                                      device=output.device).float()
                output = -(output * t_idx).sum(dim=-1, keepdim=True) # risk = -E[T]
            loss = self.compute_loss(output, batch["time"], batch["event"])
            val_loss += loss.item() * batch["event"].sum().item()

            wandb.log({"val_loss_batch": loss.item()})
            self.cindex.update(output, batch["time"], batch["event"])
            self.accumulator.update(ts_preds=output)

        val_loss /= self.val_loader.dataset.events.sum().item()  # type: ignore

        cindex, mae_nc, mae_c = self.get_metrics(training=False)

        return val_loss, cindex, mae_nc, mae_c

    def get_metrics(self, training: bool = True) -> Tuple[float, float, float]:
        """
        Get the metrics for the model.

        Args:
            training (bool): Whether the model is being evaluated on the training set.

        Returns:
            Tuple[float, float, float]: C-index, MAE uncensored, MAE censored.
        """
        if training:
            self.accumulator.fit_breslow()
            preds = self.accumulator.compute(training=True).to(self.device)
        else:
            assert isinstance(
                self.val_loader.sampler, SequentialSampler
            ), "Shuffle must be False in the validation data loader for accurate results"
            preds = self.accumulator.compute(training=False).to(self.device)

        # ensure 1D tensor before MAE / RAE
        if preds.dim() > 1:
            preds = preds.mean(dim=-1)

        cindex = self.cindex.compute()
        self.cindex.reset()

        if not training:
            # Reset the accumulator after validation
            self.accumulator.reset()

        return cindex.item(), float("nan"), float("nan")

    def save_checkpoint(self, checkpoint: str) -> None:
        """
        Save the model checkpoint, optimizer, scheduler, and the breslow estimator.

        Args:
            checkpoint (str): The path to save the checkpoint.
        """
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "accumulator": self.accumulator,
            },
            checkpoint,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the model checkpoint, optimizer, scheduler, and the breslow estimator.

        Args:
            checkpoint_path (str): The path to the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.accumulator = checkpoint["accumulator"]
        self.accumulator.to(self.device)
