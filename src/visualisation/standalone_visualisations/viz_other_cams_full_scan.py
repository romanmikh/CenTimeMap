"""
Self-contained script to visualise transformer gradients on a slice of a CT volume
Functions & classes defined here are not used downstream anywhere else in the codebase
Use: debugging & testing
"""

import torch
from torch import optim
from torch.utils.data import DataLoader
import os

from src.dataset.dummy_dataset import DummyCTDataset
from src.main import TransformerWithHead, get_trainer
from src.utils.train_utils import get_device
from src.visualisation.cam import CamInterp, cam_methods_map

import matplotlib.pyplot as plt
import numpy as np

def save_heatmap(heatmap, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    heatmap = heatmap.squeeze()  # (1, Z, Y, X) or (Z, Y, X)
    heatmap = np.max(heatmap, axis=0)
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    safe_cam_names = [
        "gradcam", "gradcam++", "ablationcam", "xgradcam"
    ]

    for use_interp_head in [True, False]:
        device = get_device()
        train_loader = DataLoader(DummyCTDataset(30), batch_size=4, shuffle=True)
        val_loader = DataLoader(DummyCTDataset(10), batch_size=4, shuffle=False)

        model = TransformerWithHead(use_interp_head=use_interp_head).to(device)
        method = "centime"
        TrainerCls, kw = get_trainer(method)
        trainer = TrainerCls(
            model=model,
            device=device,
            num_epochs=5,
            scheduler=None,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            train_loader=train_loader,
            val_loader=val_loader,
            **kw,
        )
        trainer.train()

        sample = next(iter(val_loader))
        sample = {k: v.to(device) for k, v in sample.items()}
        x = sample["img"][:1]  


        head = "interphead" if use_interp_head else "stdhead"

        for cam_name in safe_cam_names:
            print(f"\nRunning {cam_name} with {head}...")
            try:
                cam = CamInterp(cam_interp_method=cam_name, model=model)
                heatmap, _ = cam.interpret(x)
                save_path = f"results/transformer_viz/cam_methods/{cam_name}_{head}.png"
                save_heatmap(heatmap, save_path)
            except Exception as e:
                print(f"[ERROR] {cam_name} failed: {e}")
