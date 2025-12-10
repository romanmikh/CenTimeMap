"""
Base trainer class for all trainers.
"""

import inspect
from rich import print
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import yaml
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

import wandb
from src.methods.losses import (
    centime_loss,
    classical_loss,
    cox_loss,
    deephit_loss,
)
from src.metrics import MAE, CIndex
from src.utils.train_utils import BetaSchedulers, TransformerClampScheduler
from src.utils.settings import *


def get_class_args(cls: Any) -> Dict[str, Any]:
    """
    Get the arguments of a class used in the __init__ method with their used values.

    Args:
        cls (Any): The initialized object of the class.

    Returns:
        Dict[str, Any]: The arguments of the class with their values.
    """
    init_method = cls.__init__
    sig = inspect.signature(init_method)
    args = sig.parameters
    args_dict = {}
    for arg in args:
        if arg == "kwargs":
            try:
                args_dict[arg] = getattr(cls, arg)
            except AttributeError:
                continue
        if arg != "self":
            args_dict[arg] = getattr(cls, arg)
    return args_dict


class BaseTrainer(ABC):
    """
    Base trainer class for all trainers. Each trainer is a different loss function.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Union[None, optim.lr_scheduler._LRScheduler],
        device: Union[str, torch.device],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        tmax: int,
        clinical_data: bool = False,
        clip_grad_norm: Union[None, float] = None,
        exp_name: str = "",
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The model to train.
            optimizer (optim.Optimizer): The optimizer to use.
            scheduler (Union[None, optim.lr_scheduler._LRScheduler]): The learning rate
              scheduler to use.
            device (torch.device): The device to use.
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            num_epochs (int): The number of epochs to train for.
            tmax (int): The maximum possible time of death.
            clinical_data (bool): Whether to use clinical data.
            clip_grad_norm (Union[None, float]): The maximum gradient norm to clip to.
            exp_name (str): The name of the experiment.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.tmax = tmax
        self.clinical_data = clinical_data
        self.clip_grad_norm = clip_grad_norm
        self.exp_name = f"{exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.loss_fn: Optional[Callable] = None

        # metrics classes
        self.mae_uncensored = MAE(mode="uncensored").to(device)
        self.mae_censored = MAE(mode="censored").to(device)
        self.cindex = CIndex().to(device)

        # args dict to save
        self.yaml = {
            "model": self.model.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": (
                self.scheduler.__class__.__name__ if self.scheduler else None
            ),
            "num_epochs": self.num_epochs,
            "tmax": self.tmax,
            "clinical_data": self.clinical_data,
            "clip_grad_norm": self.clip_grad_norm,
            "bs": self.train_loader.batch_size,
        }

        # model args
        model_args = get_class_args(self.model)
        self.yaml.update(model_args)

    def save_args(self) -> None:
        """
        Save the arguments of the experiment as a yaml file in the experiment directory.
        """
        with open(f"runs/{self.exp_name}/args.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.yaml, f, default_flow_style=False, allow_unicode=True)

    @abstractmethod
    def compute_loss(self):
        """
        Compute the loss value and return the combined loss, the loss for uncensored,
        and the loss for censored data.
        """

    @abstractmethod
    def get_predicted_time(self, output: Tensor) -> Tensor:
        """
        Get the predicted time of death from the model output.

        Args:
            output (Tensor): The model output.

        Returns:
            Tensor: The predicted time of death.
        """

    @abstractmethod
    def train_one_epoch(
        self,
    ) -> Tuple[float, float, float, float]:
        """Train the model for one epoch."""

    @abstractmethod
    def validate_one_epoch(
        self,
    ) -> Tuple[float, float, float, float]:
        """Validate the model for one epoch."""

    def train(self) -> None:
        """Train and validate the model."""
        if self.loss_fn is cox_loss:
            metric = "cindex"
            best_metric = 0.0
        else:
            assert self.loss_fn in [
                classical_loss,
                centime_loss,
                deephit_loss,
            ], f"Loss function {self.loss_fn} not recognized."
            metric = "mae_uncensored"
            best_metric = float("inf")

        beta_scheduler  = BetaSchedulers(start_beta=BETA_HEAD_MIN, end_beta=BETA_HEAD_MAX, num_epochs=self.num_epochs, steepness=BETA_HEAD_SIGMOID_STEEPNESS)
        clamp_scheduler = TransformerClampScheduler(start_scale=0.01, end_scale=1.0, num_epochs=self.num_epochs)

        for epoch in range(self.num_epochs):
            if hasattr(self.model, "head"):
                self.model.head.beta = beta_scheduler.get_sigmoid_beta(epoch) 
                self.model.clamp_scale = clamp_scheduler.get_linear_scale(epoch)
                # self.model.clamp_scale = clamp_scheduler.get_quadratic_scale(epoch)
                # self.model.clamp_scale = clamp_scheduler.get_quadratic_scale(epoch)
                print(f"beta: {self.model.head.beta:.4f}, clamp scale: {self.model.clamp_scale:.4f} @ epoch {epoch+1}")

            t1 = time()
            tr_loss, tr_cindex, tr_mae_uncensored, tr_mae_censored = (
                self.train_one_epoch()
            )
            val_loss, val_cindex, val_mae_uncensored, val_mae_censored = (
                self.validate_one_epoch()
            )
            t2 = time()
            print(
                f"[bold cyan]Epoch {epoch + 1}/{self.num_epochs}[/]\n"

                f"[yellow]Train[/] | "
                f"Loss: [green]{tr_loss:.4f}[/]  "
                f"C-Index: [green]{tr_cindex:.4f}[/]  "
                f"MAE-U: [green]{tr_mae_uncensored:.4f}[/]  "
                f"MAE-C: [green]{tr_mae_censored:.4f}[/]\n"

                f"[yellow]Val  [/] | "
                f"Loss: [cyan]{val_loss:.4f}[/]  "
                f"C-Index: [cyan]{val_cindex:.4f}[/]  "
                f"MAE-U: [cyan]{val_mae_uncensored:.4f}[/]  "
                f"MAE-C: [cyan]{val_mae_censored:.4f}[/]\n"
                
                f":hourglass_flowing_sand: {t2 - t1:.2f}s"
            )
            wandb.log(
                {
                    "train_loss_epoch": tr_loss,
                    "train_cindex_epoch": tr_cindex,
                    "train_mae_uncensored_epoch": tr_mae_uncensored,
                    "train_mae_censored_epoch": tr_mae_censored,
                    "val_loss_epoch": val_loss,
                    "val_cindex_epoch": val_cindex,
                    "val_mae_uncensored_epoch": val_mae_uncensored,
                    "val_mae_censored_epoch": val_mae_censored,
                }
            )

            if metric == "cindex":
                if val_cindex > best_metric:
                    # print(f"New best model with C-Index: {val_cindex:.4f}")
                    best_metric = val_cindex
                    self.save_checkpoint(f"runs/{self.exp_name}/best_model.pth")
            else:
                if val_mae_uncensored < best_metric:
                    print(
                        # f"New best model with MAE Uncensored: {val_mae_uncensored:.4f}"
                    )
                    best_metric = val_mae_uncensored
                    self.save_checkpoint(f"runs/{self.exp_name}/best_model.pth")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.

        Args:
            checkpoint (str): The path to the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def save_checkpoint(self, checkpoint: str) -> None:
        """
        Save a model checkpoint.

        Args:
            checkpoint (str): The path to save the checkpoint.
        """
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (self.scheduler.state_dict() if self.scheduler else None),
            },
            checkpoint,
        )
