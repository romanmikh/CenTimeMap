"""
Self-contained script to visualise transformer gradients on a slice of a CT volume
Functions & classes defined here are not used downstream anywhere else in the codebase
Use: debugging & testing
"""

# test_cam3D_transformer_feats.py
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.dataset.dummy_dataset import DummyCTDataset
from src.visualisation.visualisers import viz_dummy_heatmap_debug
from src.main import TransformerWithHead, get_trainer
from src.utils.train_utils import get_device


if __name__ == "__main__":
    for viz_deltas in [True, False]:
        for use_interp_head in [True, False]:
            for method in ["centime", "classical", "cox", "coxmb", "deephit"]:
                DUMMY_TRAIN_SAMPLES, DUMMY_VAL_SAMPLES, BATCH, EPOCHS = 30, 10, 4, 10
                device = get_device()

                train_loader = DataLoader(DummyCTDataset(DUMMY_TRAIN_SAMPLES), batch_size=BATCH, shuffle=True)
                val_loader = DataLoader(DummyCTDataset(DUMMY_VAL_SAMPLES), batch_size=BATCH, shuffle=False)

                model = TransformerWithHead(use_interp_head=use_interp_head).to(device)
                opt   = optim.Adam(model.parameters(), lr=1e-3)
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
                heat  = "attn_map_" if viz_deltas else ""
                save = (
                    f"results/transformer_viz/transformer_feats/full_scan/"
                    f"backbone_{heat}{method}_{head}.png"
                )
                viz_dummy_heatmap_debug(model, val_loader, 
                                save_path=save, 
                                show_head_feats=False, 
                                viz_deltas=viz_deltas,
                                device=device)
