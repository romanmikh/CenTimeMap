# pylint: disable=no-member

import matplotlib.pyplot as plt, numpy as np
import torch
from torch import nn
from rich import print
from typing import Union, Tuple
from src.methods.losses import (
    discretized_gaussian,
    get_mean_prediction,
    get_survival_function,
)
from src.utils.settings import *


class StandardHead(nn.Module):
    """Standard head. Pools tokens without considering cubelets."""

    def __init__(self, DIM_IN: int, tmax: int, var: float, **_):
        super().__init__()
        self.var = var
        self.tmax = tmax
        self.fc = nn.Linear(DIM_IN, 1)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Input tensor. Shape (B, nZ, nY, nX, D)

        Returns:
            Tensor of histograms of expected survival times, shape (B, tmax).
        """
        x = x.view(x.size(0), -1, x.size(-1))  # [B, N, D]
        x = x.mean(dim=1)  # [B, D]
        loc = self.fc(x)  # [B, 1]
        var = torch.ones_like(loc) * self.var  # [B, 1]
        p_tx = discretized_gaussian(  # [B, Tmax]
            loc.view(-1, 1), var.view(-1, 1), self.tmax, return_prob=True
        )
        return p_tx


class InterpretabilityHead(nn.Module):
    """Weighs cubelets by importance and marginalises to compute p_tx"""

    def __init__(
        self, dim: int, tmax: int, var: float, beta: float, hidden: int, **kwargs
    ):
        super().__init__(**kwargs)
        self.var = var
        self.tmax = tmax
        self.hidden = hidden
        self.beta = float(beta)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),  # understand why standardising intra-patch helps
            nn.Linear(dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )

        # global bias to calibrate loc
        self.loc_bias = nn.Parameter(
            torch.tensor(LOC_BIAS)
        )  # check why bias changes so little during training
        self.softmin = nn.Softmin(dim=-1)
        # self.beta = nn.Parameter(torch.tensor(self.beta))         # consider beta per cubelet in the future instead of scalar
        # self.register_buffer("beta", torch.tensor(self.beta, dtype=torch.float32))

    def forward(
        self, x, *, return_loc: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor. Shape (B, nZ, nY, nX, D)

        Returns:
            p_tx: Tensor of histograms of expected survival times marginalised over all cubelets, shape (B, tmax).
        """
        x = x.view(
            x.size(0), -1, x.size(-1)
        )  # [B, N, D]     B patients, each with N cubelets, each with D features
        loc = (
            self.mlp(x).squeeze(-1) + self.loc_bias
        )  # [B, N]        expected time to death in months per cubelet
        center = loc.detach().median(dim=-1, keepdim=True).values
        p_i = self.softmin(
            self.beta * (loc - center)
        )  # necessary with large LOC_BIAS, softmin sensitive to absolute values
        var = torch.ones_like(loc) * self.var  # [B, N]
        p_tix = discretized_gaussian(
            loc.view(-1, 1),
            var.view(-1, 1),
            self.tmax,
            return_prob=True,  # TODO check if using log space is more stable & efficient
        )  # [B*N , Tmax]
        p_tix = p_tix.view(x.size(0), -1, self.tmax)  # [B, N, Tmax]
        weighted_p_tix = p_tix * p_i.unsqueeze(-1)  # [B, N, Tmax]
        weighted_p_tx = weighted_p_tix.sum(dim=1)  # [B, Tmax]

        # [DEBUG]
        # print(f"      x: min {     x.min().item():.1f}, max {     x.max().item():.1f}, mean {     x.mean().item():.1f}, std {     x.std().item():.1f}, median {     x.median().item():.1f}, frac_neg {((     x <= 0).sum().item() /      x.numel()):.3f},      x sum: {     x.sum():.3f}")
        # print(f"    loc: min {   loc.min().item():.1f}, max {   loc.max().item():.1f}, mean {   loc.mean().item():.1f}, std {   loc.std().item():.1f}, median {   loc.median().item():.1f}, frac_neg {((   loc <= 0).sum().item() /    loc.numel()):.3f},    loc sum: {   loc.sum():.3f}")
        # print(f" center: {center.mean().item():.1f}")
        # print(f"    p_i: min {    p_i.min().item():.4f}, max {    p_i.max().item():.4f}, mean {    p_i.mean().item():.4f}, std {    p_i.std().item():.4f}, median {    p_i.median().item():.4f}, frac_neg {((    p_i <= 0).sum().item() /     p_i.numel()):.3f},     p_i sum: {    p_i.sum():.3f}")
        # print(f"  p_tix: min {  p_tix.min().item():.4f}, max {  p_tix.max().item():.4f}, mean {  p_tix.mean().item():.4f}, std {  p_tix.std().item():.4f}, median {  p_tix.median().item():.4f}, frac_neg {((  p_tix <= 0).sum().item() /   p_tix.numel()):.3f},   p_tix sum: {  p_tix.sum():.3f}")
        # print(f"wtp_tix: min {weighted_p_tix.min().item():.4f}, max {weighted_p_tix.max().item():.4f}, mean {weighted_p_tix.mean().item():.4f}, std {weighted_p_tix.std().item():.4f}, median {weighted_p_tix.median().item():.4f}, frac_neg {((weighted_p_tix <= 0).sum().item() / weighted_p_tix.numel()):.3f}, weighted_p_tix sum: {weighted_p_tix.sum():.3f}")
        # print(f" wtp_tx: min { weighted_p_tx.min().item():.4f}, max { weighted_p_tx.max().item():.4f}, mean { weighted_p_tx.mean().item():.4f}, std { weighted_p_tx.std().item():.4f}, median { weighted_p_tx.median().item():.4f}, frac_neg {(( weighted_p_tx <= 0).sum().item() /  weighted_p_tx.numel()):.3f},  wtp_tx sum: { weighted_p_tx.sum():.3f}")
        # print(f"weighted_p_tx: {weighted_p_tx}")
        # print(f"self.loc_bias: {self.loc_bias.item():.4f}")

        # [DEBUG] plot loc distribution
        # v = loc.detach().flatten().cpu().numpy()
        # plt.figure(); plt.hist(v, bins=100, density=True); plt.xlabel("loc"); plt.ylabel("pdf"); plt.tight_layout(); plt.savefig("results/loc_freq_pdf.pdf"); plt.close()

        expected_from_hist = get_mean_prediction(weighted_p_tx, self.tmax)  # [B,]
        print(f"Prediction: {expected_from_hist} months, var:  {self.var.item():.4f}")
        # expected_from_surv_function = get_survival_function(weighted_p_tx).sum(axis=-1)  # [B,]
        # print("E[T] from wptx histogram:", expected_from_hist, ", E[T] from wptxsurv_function: ", expected_from_surv_function, "should be approx equal")

        return (weighted_p_tx, loc) if return_loc else weighted_p_tx
