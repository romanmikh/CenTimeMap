"""
Self-contained script to visualise transformer gradients on a slice of a CT volume
Functions & classes defined here are not used downstream anywhere else in the codebase
Use: debugging & testing
"""

import torch
from torch.utils.data import DataLoader
from torch import optim

from src.dataset.dummy_dataset import DummyCTDataset
from src.visualisation.visualisers import viz_feats_slice
from src.main import TransformerWithHead, get_trainer
from src.utils.train_utils import get_device

for use_interp_head in [True, False]:
    for method in ["centime","classical","cox","coxmb","deephit"]:

        DUMMY_TRAIN_SAMPLES, DUMMY_VAL_SAMPLES, BATCH, EPOCHS = 30, 10, 4, 10
        device = get_device()

        train_loader = DataLoader(DummyCTDataset(DUMMY_TRAIN_SAMPLES,), batch_size=BATCH, shuffle=True)
        val_loader   = DataLoader(DummyCTDataset(DUMMY_VAL_SAMPLES,), batch_size=BATCH, shuffle=False)

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

        # sample to visualise
        vol = DummyCTDataset(1, feat_frac=1.0)[0]["img"][0]
        mid = vol.shape[0] // 2
        ct_slice = vol[mid]                                             # (H, W)

        model.eval()
        with torch.no_grad():
            feats = model.backbone(vol.unsqueeze(0).unsqueeze(0).to(device))  # (1, Z',Y',X',D)

        head  = "interphead" if use_interp_head else "stdhead"
        save_path = (
            f"results/transformer_viz/transformer_feats/slice/"
            f"backbone_{method}_{head}.png"
        )
        viz_feats_slice(feats, ct_slice, save_path=save_path)

        