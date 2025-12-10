"""
Self-contained script to visualise transformer gradients on a slice of a CT volume
Functions & classes defined here are not used downstream anywhere else in the codebase
Use: debugging & testing
"""

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.dataset.dummy_dataset import DummyCTDataset
from src.visualisation.visualisers import viz_dummy_heatmap_debug
from src.main import TransformerWithHead, get_trainer
from src.utils.train_utils import get_device


if __name__ == "__main__":
    for target in ["backbone", "CNN"]:
        for use_interp_head in [True, False]:
            for method in ["centime", "classical", "cox", "coxmb", "deephit"]:
                TRAIN, VAL, BATCH, EPOCHS = 30, 10, 4, 5
                device = get_device()

                train_loader = DataLoader(DummyCTDataset(TRAIN), batch_size=BATCH, shuffle=True)
                val_loader = DataLoader(DummyCTDataset(VAL), batch_size=BATCH, shuffle=False)

                model = TransformerWithHead(use_interp_head=use_interp_head).to(device)
                opt = optim.Adam(model.parameters(), lr=1e-3)
                TrainerCls, kw = get_trainer(method)

                trainer = TrainerCls(
                    model=model,
                    device=device,
                    num_epochs=EPOCHS,
                    scheduler=None,
                    optimizer=opt,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    **kw,
                )
                trainer.train()

                head = "interphead" if use_interp_head else "stdhead"
                save = (
                    "results/transformer_viz/transformer_grads/full_scan/"
                    f"{target}_{method}_{head}.png"
                )

                viz_dummy_heatmap_debug(
                    model,
                    val_loader,
                    save_path=save,
                    viz_grads=True,   # Grad-CAM mode
                    target=target,    # "CNN" or "backbone"
                    device=device,
                )
