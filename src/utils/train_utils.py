import sys
import math
import torch
import random
import numpy as np
from pathlib import Path
from src.methods import *


class TransformerClampScheduler:
    def __init__(self, start_scale=0.01, end_scale=1.0, num_epochs=10):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.num_epochs = num_epochs

    def get_linear_scale(self, epoch):
        """Linearly scale the clamp over num_epochs"""
        return self.start_scale + (self.end_scale - self.start_scale) * float(epoch) / (self.num_epochs - 1)

    def get_quadratic_scale(self, epoch):  
        """Quadratically scale the clamp over num_epochs"""
        ep2 = float(epoch) ** 2
        eps2 = float(self.num_epochs - 1) ** 2
        return self.start_scale + (self.end_scale - self.start_scale) * ep2 / eps2
    
    def get_quartic_scale(self, epoch): # seems to encourage sphere inversion issue, do not use
        """Quartically scale the clamp over num_epochs"""
        ep4 = float(epoch) ** 4
        eps4 = float(self.num_epochs - 1) ** 4
        return self.start_scale + (self.end_scale - self.start_scale) * ep4 / eps4
    

class BetaSchedulers:
    def __init__(self, start_beta=0.1, end_beta=5.0, num_epochs=100, steepness=10.0):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.num_epochs = num_epochs
        self.steepness = steepness

    def get_sigmoid_beta(self, epoch): # seems to work better than linear, experiment with params
        """Sigmoidally scale beta over num_epochs"""
        epochs_norm = 2 * (epoch / (self.num_epochs - 1)) - 1
        sig = 1 / (1 + math.exp(-self.steepness * epochs_norm))
        return self.start_beta + (self.end_beta - self.start_beta) * sig

    def get_linear_beta(self, epoch):
        """Linearly scale beta over num_epochs"""
        return self.start_beta + (self.end_beta - self.start_beta) * float(epoch) / (self.num_epochs - 1)


def detect_data():
    for i, a in enumerate(sys.argv):
        if a == "--data" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--data="):
            return a.split("=", 1)[1]
    return None


def is_using_dummydataset():
    data = detect_data()
    if data and data.startswith("dummy"):
        return True
    return False


def load_checkpoint(model, device):
    """Load the latest checkpoint if available"""
    ckpt_candidates = sorted(
        Path("runs").glob("CenTime_*/best_model.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    ckpt_path = ckpt_candidates[0]
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.clamp_scale = 1.0
    return ckpt_path


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_grad_norm(model):
    """Get the L2 norm of the gradients, for debugging purposes."""
    total_norm = 0
    for par in model.parameters():
        if par.grad is not None:
            param_norm = par.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
