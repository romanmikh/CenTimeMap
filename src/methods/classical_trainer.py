# pylint: disable=super-init-not-called, non-parent-init-called
import os

import wandb
from src.methods import BaseTrainer, CenTimeTrainer
from src.methods.losses import classical_loss


class ClassicalTrainer(CenTimeTrainer):
    """
    Trainer for the classical loss. Same stuff as CenTime, but with a different loss
    function and naming. Not a very clean way to do it.
    """

    def __init__(
        self,
        distribution: str = "discretized_gaussian",
        variance: float = 144.0,
        alpha_cens: float = 1.0,
        **kwargs,
    ):
        BaseTrainer.__init__(self, **kwargs)
        self.distribution = distribution
        self.variance = variance
        self.alpha_cens = alpha_cens
        self.loss_fn = classical_loss
        self.exp_name = "Classical" + self.exp_name
        
        if wandb.run is not None:
            wandb.finish()

        wandb.init(project="centime", name=self.exp_name)
        os.makedirs(f"runs/{self.exp_name}")
        self.yaml.update(
            {
                "loss_fn": "classical_loss",
                "distribution": self.distribution,
                "variance": self.variance,
                "alpha_cens": self.alpha_cens,
                "exp_name": self.exp_name,
            }
        )
        self.save_args()
