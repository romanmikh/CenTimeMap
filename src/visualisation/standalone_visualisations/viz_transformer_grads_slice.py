"""
Self-contained script to visualise transformer gradients on a slice of a CT volume
Functions & classes defined here are not used downstream anywhere else in the codebase
Use: debugging & testing
"""

import torch
from torch import optim
from pytorch_grad_cam import GradCAM
from torch.utils.data import DataLoader

from src.dataset.dummy_dataset import DummyCTDataset
from src.visualisation.visualisers import viz_grad_slice, get_target_CNN_layer, reshape_transform
from src.main import TransformerWithHead, get_trainer
from src.utils.train_utils import get_device


if __name__ == "__main__":
    for target in ["backbone", "CNN"]:
        for use_interp_head in [True, False]:
            for method in ["centime", "classical", "cox", "coxmb", "deephit"]:
                TRAIN, VAL, BATCH, EPOCHS = 30, 10, 4, 10 
                device = get_device()

                train_loader = DataLoader(DummyCTDataset(TRAIN), batch_size=BATCH, shuffle=True)
                val_loader   = DataLoader(DummyCTDataset(VAL),   batch_size=BATCH, shuffle=False)

                model = TransformerWithHead(use_interp_head=use_interp_head).to(device)
                opt   = optim.Adam(model.parameters(), lr=1e-3)
                TrainerCls, kw = get_trainer(method)
                trainer = TrainerCls(model=model, 
                    device=device, 
                    num_epochs=EPOCHS,
                    scheduler=None, 
                    optimizer=opt,
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    **kw
                )
                trainer.train()

                # sample to visualise
                vol      = DummyCTDataset(1, feat_frac=1.0)[0]["img"][0]      # (Z,Y,X)
                ct_slice = vol[vol.shape[0] // 2]                              # (Y,X)

                model.eval()
                if target == "CNN":
                    target_layer = get_target_CNN_layer(model)
                    reshape = None
                elif target == "backbone":
                    target_layer = model.backbone
                    reshape = reshape_transform
                
                # print number of tokens coming from the backbone
                print(f"Number of tokens from the backbone: {target_layer.__sizeof__()}")
                
                cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape)
                heat = cam(input_tensor=vol.unsqueeze(0).unsqueeze(0).to(device), targets=[lambda y: y.sum()])[0]   # (Z',Y',X')

                head = "interphead" if use_interp_head else "stdhead"
                out  = (f"results/transformer_viz/transformer_grads/slice/"
                        f"{target}_grads_{method}_{head}.png")
                viz_grad_slice(heat, ct_slice, save_path=out)
