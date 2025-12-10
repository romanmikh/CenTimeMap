import torch
import h5py
from torch import Tensor, nn
import argparse
import numpy as np
import pandas as pd
from rich import print
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

from src.methods import *
from src.models import StandardHead, InterpretabilityHead
from src.dataset import OSICDataset
from src.transformer import TransformerBackbone
from src.dataset.dummy_dataset import dummy_dataset_manager
from src.utils.settings import *
from src.utils.argparser import parse_input
from src.utils.train_utils import get_device, load_checkpoint
from src.visualisation.visualisers import _compute_heat, viz_all
from src.visualisation.utils import _scale_overlapped_heatmap


def get_trainer(method: str, variance: float):
    method = method.lower()
    mapping = {
        "centime": (
            CenTimeTrainer,
            dict(tmax=TMAX, variance=variance, alpha_cens=ALPHA_CENS),
        ),
        "classical": (
            ClassicalTrainer,
            dict(tmax=TMAX, variance=variance, alpha_cens=ALPHA_CENS),
        ),
        "cox": (CoxTrainer, dict(tmax=TMAX)),
        "coxmb": (CoxMBTrainer, dict(tmax=TMAX)),
        "deephit": (DeepHitTrainer, dict(tmax=TMAX)),
    }
    if method not in mapping:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(mapping)}")
    return mapping[method]


class TransformerWithHead(nn.Module):
    """Wrapper connecting TransformerBackbone with standard or interpretability heads."""

    def __init__(self, args=None, var=None):
        super().__init__()
        self.args = args
        # with torch.no_grad():
        self.backbone = TransformerBackbone(
            dim=DIM_IN,
            codebook_size=CODEBOOK_SIZE,
            depth=NUM_BLOCKS,
            image_size=(CT_SIZE_D, CT_SIZE_W, CT_SIZE_H),
            patch_size=(PATCH_H, PATCH_W, PATCH_D),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        # https://arxiv.org/abs/1709.07871
        # MLP to process clinical data into FiLM parameters
        # simply: shape the MLP into two tensors, which then scale and shift the transformer features
        self.clin_mlp = nn.Sequential(
            nn.Linear(30, DIM_IN),
            nn.ReLU(inplace=True),
            nn.Linear(DIM_IN, 2 * DIM_IN),
        )

        # head_kwargs = dict(dim=DIM_IN, tmax=TMAX, var=var)
        self.var = nn.Parameter(
            torch.tensor(var if var is not None else VAR_INIT, dtype=torch.float32)
        )
        # self.log_var = nn.Parameter(torch.tensor(np.log(VAR_INIT), dtype=torch.float))
        head_kwargs = dict(dim=DIM_IN, tmax=TMAX, var=self.var)
        if self.args.head == "interp":
            head_kwargs.update(beta=BETA_HEAD, hidden=HEAD_HIDDEN)
            self.head = InterpretabilityHead(**head_kwargs)
        else:
            self.head = StandardHead(**head_kwargs)

    def forward(self, img, clinical_data=None, mask=None) -> torch.Tensor:
        """
        Args:
            img: CT scan input tensor (B, 1, Z, Y, X)
            clinical_data: (B, n_cols) or None

        Returns:
            p_tx: Tensor of histograms of expected survival times, shape (B, tmax)
        """
        feats = self.backbone(
            img, mask
        )  # ((B, Z', Y', X', D), optional (B, Z', Y', X'))
        for p in self.backbone.parameters():
            p.requires_grad = False

        if isinstance(feats, tuple):
            feats = feats[0]

        # clamp transformer output to stabilise training
        feats = torch.clamp(
            feats,
            min=-feats.max() * self.clamp_scale,
            max=feats.max() * self.clamp_scale,
        )
        print(
            f"Feature min / max after clamp: {feats.min().item():.4f} / {feats.max().item():.4f}"
        )

        if clinical_data:
            # FiLM in 1 equation (as a starting point): feats' = (1 + gamma) * feats + beta
            gamma_beta = self.clin_mlp(clinical_data)  # (B, 2D)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # (B, D), (B, D)

            # reshape to Z',Y',X' dims
            gamma = gamma[:, None, None, None, :]  # (B,1,1,1,D)
            beta = beta[:, None, None, None, :]  # (B,1,1,1,D)
            feats = feats * (1.0 + gamma) + beta  # (B,Z',Y',X',D)
            # TODO adjust weight of clinical data. Scale both terms / regularise / initialise non-randomly

        p_tx = self.head(feats)
        return p_tx

    def main():
        args = parse_input()
        print(f"[yellow][!] Using {args.data} dataset[/yellow]")
        initial_var = float(VAR_INIT)

        if args.data == "osic":
            train_dataset = OSICDataset(
                data_path=OSIC_DATA_PATH,
                clinical_data_path="/SAN/medic/IPF/centime_map_ms/clinical_data_all_correct_dlco_thesis_july24_region.csv",
                split="train",
                segment=False,
                fold=1,
                num_patients=-1,
                p_uncens=1.0,
                use_lung_mask=True,
                use_clinical_data=False,
                clinical_normalizer=None,
                clinical_encoder=None,
                load_imgs=True,
                tmax=TMAX,
                n_dummy_states=0,
            )

            val_dataset = OSICDataset(
                data_path=OSIC_DATA_PATH,
                clinical_data_path="/SAN/medic/IPF/centime_map_ms/clinical_data_all_correct_dlco_thesis_july24_region.csv",
                split="val",
                segment=False,
                fold=1,
                num_patients=-1,
                p_uncens=1.0,
                use_lung_mask=True,
                use_clinical_data=False,
                clinical_normalizer=None,
                clinical_encoder=None,
                load_imgs=True,
                tmax=TMAX,
                n_dummy_states=0,
            )
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        else:
            dummy_train = dummy_dataset_manager(
                n_samples=DUMMY_TRAIN_SAMPLES, complexity=int(args.data[-1])
            )
            dummy_val = dummy_dataset_manager(
                n_samples=DUMMY_VAL_SAMPLES, complexity=int(args.data[-1])
            )
            train_loader = DataLoader(dummy_train, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(dummy_val, batch_size=BATCH_SIZE, shuffle=False)

        device = get_device()
        model = TransformerWithHead(args, initial_var)

        # load checkpoint if requested
        loaded = False
        if args.load_ckpt:
            try:
                ckpt_path = load_checkpoint(model, device)
                print(f"[green][✓] Loaded checkpoint: {ckpt_path}[/green]")
                loaded = True
            except Exception as e:
                print(
                    f"[yellow][!] Failed to load checkpoint: {e}. Proceeding to train from scratch.[/yellow]"
                )

        if not loaded:
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            # scheduler = ExponentialLR(optimizer, gamma=0.95)
            TrainerClass, extra_kwargs = get_trainer(args.method, variance=initial_var)
            trainer = TrainerClass(
                model=model,
                device=device,
                num_epochs=EPOCHS,
                scheduler=SCHEDULER,
                optimizer=optimizer,
                val_loader=val_loader,
                train_loader=train_loader,
                clinical_data=False,  # set to True to use clinical data
                use_lung_mask=True,  # set to True to use lung masks
                **extra_kwargs,
            )
            print(f"[bold cyan]\n*** Running {TrainerClass.__name__} ***\n[/bold cyan]")
            trainer.train()

        model.to(device).eval()
        viz_all(model=model, val_loader=val_loader, args=args)

        # save CT, heatmap & overlay as .h5 files
        sample = next(iter(val_loader))
        ct_gpu = sample["img"][:1].to(device)  # (1,1,Z,Y,X) tensor
        ct_vol = ct_gpu[0, 0].cpu().numpy()  # (Z,Y,X) numpy

        # heatmap
        heat, _, _ = _compute_heat("head_feats", model, ct_gpu)
        heat_vol = heat.cpu().numpy()  # (Z,Y,X) numpy

        # overlay
        heat_scaled = _scale_overlapped_heatmap(
            token_heat=heat_vol, ct_shape=(CT_SIZE_D, CT_SIZE_H, CT_SIZE_W)
        )
        heat_vol = heat_scaled * ct_vol
        overlay = ct_vol * heat_vol  # (Z,Y,X) numpy

        save_h5_results(
            ct_vol=ct_vol,
            heat_vol=heat_vol,
            overlay=overlay,
            save_path="patient_abcd1234.h5",
        )


def save_h5_results(ct_vol, heat_vol, overlay, save_path="default.h5"):
    """Save CT, heatmap & overlay together in a single .h5"""
    with h5py.File(save_path, "w") as f:
        f.create_dataset("CT_scan", data=ct_vol)  # (Z,Y,X)
        f.create_dataset("CenTimeMap", data=heat_vol)  # (Z,Y,X)
        f.create_dataset("Overlay", data=overlay)  # (Z,Y,X)


if __name__ == "__main__":
    TransformerWithHead.main()
