# pylint: disable=no-member, arguments-differ
import os
from typing import Tuple, no_type_check
from rich import print

import torch
from torch import Tensor, nn
import random

import wandb
from src.methods import BaseTrainer
from src.methods.losses import centime_loss, get_mean_prediction, get_probs
from src.utils.train_utils import get_grad_norm

def occlude_patch(volume, center, patch_size=(8, 8, 8), value=0.0):
    occluded = volume.clone()
    _, D, H, W = occluded.shape
    x, y, z = center
    dx, dy, dz = patch_size

    x_min = max(x - dx // 2, 0)
    x_max = min(x + dx // 2, D)
    y_min = max(y - dy // 2, 0)
    y_max = min(y + dy // 2, H)
    z_min = max(z - dz // 2, 0)
    z_max = min(z + dz // 2, W)

    occluded[0, x_min:x_max, y_min:y_max, z_min:z_max] = value
    return occluded


class CenTimeTrainer(BaseTrainer):
    """
    Trainer for CenTime.
    """

    def __init__(
        self,
        alpha_cens,
        variance,
        distribution: str = "discretized_gaussian",
        use_lung_mask: bool = False,
        **kwargs,
    ):
        self.apply_occlusion_aug = kwargs.pop("apply_occlusion_aug", False)
        super().__init__(**kwargs)
        self.distribution = distribution
        # self.variance = variance
        self.variance = self.model.var
        self.alpha_cens = alpha_cens
        self.use_lung_mask = use_lung_mask
        self.loss_fn = centime_loss
        self.exp_name = "CenTime" + self.exp_name

        if wandb.run is not None:
            wandb.finish()
        wandb.init(project="centime", name=self.exp_name)
        os.makedirs(f"runs/{self.exp_name}")
        self.yaml.update(
            {
                "loss_fn": "centime_loss",
                "distribution": self.distribution,
                "variance": self.variance,
                "alpha_cens": self.alpha_cens,
                "exp_name": self.exp_name,
            }
        )
        self.save_args()

    @no_type_check
    def compute_loss(
        self, surv_dists: Tensor, target: Tensor, event: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the CenTime loss.
        """
        loss_censored, loss_uncensored = self.loss_fn(
            surv_dists, event, target, tmax=self.tmax, distribution=self.distribution
        )
        loss = self.alpha_cens * loss_censored + loss_uncensored
        return loss, loss_uncensored, loss_censored

    def get_predicted_time(self, loc: Tensor) -> Tensor:
        var = torch.ones_like(loc) * self.variance
        distributions = get_probs(
            loc, var, tmax=self.tmax, distribution=self.distribution
        )
        pred = get_mean_prediction(distributions, tmax=self.tmax)
        return pred

    def apply_random_occlusion(self, volume, num_patches=5, patch_size=(8, 8, 8), drop_prob=0.5):
        """
        Randomly occludes patches in a 3D volume with a given dropout probability.
        """
        _, D, H, W = volume.shape
        for _ in range(num_patches):
            if random.random() < drop_prob:
                center = (
                    random.randint(patch_size[0] // 2, D - patch_size[0] // 2),
                    random.randint(patch_size[1] // 2, H - patch_size[1] // 2),
                    random.randint(patch_size[2] // 2, W - patch_size[2] // 2),
                )
                volume = occlude_patch(volume, center, patch_size, value=0.0)
        return volume

    def train_one_epoch(
        self,
    ) -> Tuple[float, float, float, float]:
        """
        Train the model for one epoch.
        """
        self.model.train()
        train_loss = 0.0
        for batch in self.train_loader:
            # batch = {k: v.to(self.device) for k, v in batch.items()}
            batch = {k: v.to(self.device) for k, v in batch.items() if type(v) is not list}
            # print if sample has sphere or not 
            # print(f"[blue]Batch img shape: {batch['event']}[/blue]")
            img = batch["img"]
            # mask = batch["lung_mask"] if self.use_lung_mask else None
            # Apply random occlusion augmentation if enabled
            if self.apply_occlusion_aug:
                for i in range(img.shape[0]):
                    img[i] = self.apply_random_occlusion(img[i])
            clinical_data = batch["clinical_data"] if self.clinical_data else None
            self.optimizer.zero_grad()
            surv_dists  = self.model(img, clinical_data)
            assert torch.all(surv_dists >= 0), "Survival distributions must be non-negative."
            loss, loss_uncensored, loss_censored = self.compute_loss(
                surv_dists, batch["time"], batch["event"]
            )
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
            # print(f"[bold red]Train loss: {train_loss:.4f}[/bold red]")
            # print(f"[bold green]loss.item(): {loss.item()}[/bold green]")
            # print(f"[bold green]len(batch[time]): {len(batch["time"])}[/bold green]")

            wandb.log(
                {
                    "train_loss_batch": loss.item(),
                    "train_loss_uncensored_batch": loss_uncensored.item(),
                    "train_loss_censored_batch": loss_censored.item(),
                    "grad_norm": grad_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            pred_time = get_mean_prediction(surv_dists, tmax=self.tmax).view(-1)
            self.mae_uncensored.update(pred_time, batch["time"], batch["event"])
            self.mae_censored.update(pred_time, batch["time"], batch["event"])
            # print("VAL pred_time :", pred_time[:5])
            # print("VAL time      :", batch["time"][:5])
            # print("VAL event     :", batch["event"][:5])

            self.cindex.update(-pred_time, batch["time"], batch["event"])

        train_loss /= len(self.train_loader.dataset)  # type: ignore

        mae_nc = self.mae_uncensored.compute().item()
        self.mae_uncensored.reset()
        mae_c = self.mae_censored.compute().item()
        self.mae_censored.reset()
        cindex = self.cindex.compute().item()
        self.cindex.reset()

        return train_loss, cindex, mae_nc, mae_c

    def validate_one_epoch(
        self,
    ) -> Tuple[float, float, float, float]:
        """
        Validate the model for one epoch.
        """
        self.model.eval()
        val_loss = 0.0
        for batch in self.val_loader:
            # batch = {k: v.to(self.device) for k, v in batch.items()}
            batch = {k: v.to(self.device) for k, v in batch.items() if type(v) is not list}
            img = batch["img"]
            # mask = batch["lung_mask"] if self.use_lung_mask else None
            clinical_data = batch["clinical_data"] if self.clinical_data else None
            surv_dists = self.model(img, clinical_data)
            assert torch.all(surv_dists >= 0), "Survival distributions must be non-negative."

            # debugging raw cubelet predictions duirng training
            # with torch.no_grad():                               
            #     feats = self.model.backbone(img)                        # (B,Z',Y',X',D)
            #     _, loc_batch = self.model.head(feats, return_loc=True)  # _, loc (B,N)

            #     for i in range(loc_batch.size(0)):                     
            #         loc_i = loc_batch[i]                                # (N,)
            #         print("raw cubelet predictions:\n", loc_i.flatten())
            #         print("loc min / max:", 
            #             float(loc_i.min()), float(loc_i.max()))
            #         print("sample time / event", 
            #             float(batch['time'][i]), float(batch['event'][i]))
            #         print("-"*70)

            loss, loss_uncensored, loss_censored = self.compute_loss(
                surv_dists, batch["time"], batch["event"]
            )
            val_loss += loss.item() * len(batch["time"])

            wandb.log(
                {
                    "val_loss_batch": loss.item(),
                    "val_loss_uncensored_batch": loss_uncensored.item(),
                    "val_loss_censored_batch": loss_censored.item(),
                }
            )
            pred_time = get_mean_prediction(surv_dists, tmax=self.tmax).view(-1)
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
