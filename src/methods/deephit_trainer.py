# pylint: disable=arguments-differ
import os
import wandb
import torch
from torch import Tensor, nn
from typing import Tuple, no_type_check

from src.methods import BaseTrainer
from src.methods.losses import deephit_loss
from src.utils.train_utils import get_grad_norm


class DeepHitTrainer(BaseTrainer):
    """
    Trainer for DeepHit.
    """

    def __init__(self, ranking: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ranking = ranking
        self.loss_fn = deephit_loss
        self.exp_name = "DeepHit" + self.exp_name

        if wandb.run is not None:
            wandb.finish()

        wandb.init(project="centime", name=self.exp_name)
        os.makedirs(f"runs/{self.exp_name}")
        self.yaml.update(
            {
                "ranking": self.ranking,
                "loss_fn": "deephit_loss",
                "exp_name": self.exp_name,
            }
        )
        self.save_args()

    @no_type_check
    def compute_loss(self, output: Tensor, target: Tensor, event: Tensor) -> Tensor:
        """
        Compute the DeepHit loss.
        """
        event = event.view(-1, 1)
        target = target.view(-1, 1)
        return self.loss_fn(output, event, target, ranking=self.ranking)

    def get_predicted_time(self, output: Tensor) -> Tensor:
        """
        Get the predicted time of death from the model output.
        """
        return output.argmax(dim=1).reshape(-1).long()
    
    # TODO: a partial solution to noisy metrics, look further into this
    # def get_predicted_time(self, output: Tensor) -> Tensor:
    #     """Use E[T] instead of arg-max for smoother MAE/C-index."""
    #     bins = torch.arange(1, self.tmax + 1, device=output.device).float()
    #     return (output * bins).sum(dim=1)          # shape (B,)


    def train_one_epoch(
        self,
    ) -> Tuple[float, float, float, float]:
        """
        Train the model for one epoch. Cox models require a different function so this
        will be overridden.
        """
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            img = batch["img"]
            clinical_data = batch["clinical_data"] if self.clinical_data else None
            self.optimizer.zero_grad()
            output = self.model(img, clinical_data).softmax(dim=1)
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

            train_loss += loss.item() * len(batch["time"])

            wandb.log(
                {
                    "train_loss_batch": loss.item(),
                    "grad_norm": grad_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            pred_time = self.get_predicted_time(output)
            self.mae_uncensored.update(pred_time, batch["time"], batch["event"])
            self.mae_censored.update(pred_time, batch["time"], batch["event"])
            self.cindex.update(-pred_time, batch["time"], batch["event"])

        train_loss /= len(self.train_loader.dataset)  # type: ignore

        mae_nc = self.mae_uncensored.compute().item()
        self.mae_uncensored.reset()
        mae_c = self.mae_censored.compute().item()
        self.mae_censored.reset()
        cindex = self.cindex.compute().item()
        self.cindex.reset()

        return train_loss, cindex, mae_nc, mae_c

    def validate_one_epoch(self) -> Tuple[float, float, float, float]:
        """
        Validate the model for one epoch using the DeepHit loss.
        """
        self.model.eval()
        val_loss = 0.0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            img = batch["img"]
            clinical_data = batch["clinical_data"] if self.clinical_data else None
            output = self.model(img, clinical_data).softmax(dim=1)
            loss = self.compute_loss(output, batch["time"], batch["event"])
            val_loss += loss.item() * len(batch["time"])

            wandb.log({"val_loss_batch": loss.item()})

            pred_time = self.get_predicted_time(output)
            self.mae_uncensored.update(pred_time, batch["time"], batch["event"])
            self.mae_censored.update(pred_time, batch["time"], batch["event"])
            self.cindex.update(-pred_time, batch["time"], batch["event"])

        val_loss /= len(self.val_loader.dataset)  # type: ignore

        mae_nc = self.mae_uncensored.compute().item()
        self.mae_uncensored.reset()
        mae_c = self.mae_censored.compute().item()
        self.mae_censored.reset()
        cindex = self.cindex.compute().item()
        self.cindex.reset()

        return val_loss, cindex, mae_nc, mae_c
