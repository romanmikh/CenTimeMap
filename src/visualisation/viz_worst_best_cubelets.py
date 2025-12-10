from src.utils.train_utils import get_device
import torch
import numpy as np
from rich import print
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.settings import *


def viz_worst_best_cubelets(
    model,
    loader,
    num_cubelets: int = 5,
    device=get_device(),
    save_path: str | Path = "results/cubelets_worst_best.png",
):
    """
    Show the axial mid-slice of the cubelets whose predicted mean survival time 
    is lowest and highest in a 2xN grid.

    Should only be used with InterpretabilityHead
    """
    model.eval().to(device)

    # one validation volume 
    batch = next(iter(loader))
    ct_gpu = batch["img"][:1].to(device)              # (1,1,Z,Y,X)
    ct_vol = ct_gpu[0, 0].cpu().numpy()               # (Z,Y,X)

    with torch.no_grad():
        feats = model.backbone(ct_gpu)[0]             # (1,nZ,nY,nX,D)
        _, loc  = model.head(feats, return_loc=True)  # loc: (1,N)
    loc = loc[0]                                      # (N,)

    nZ, nY, nX = feats.shape[1:4]
    patch_dims = (PATCH_D, PATCH_H, PATCH_W)
    stride     = tuple(p // PATCH_OVERLAP for p in patch_dims)

    # indices of the extreme cubelets 
    lows  = torch.topk(loc, num_cubelets, largest=False).indices
    highs = torch.topk(loc, num_cubelets, largest=True ).indices
    # TODO: locs can be negative right now, they're normalised, fix this

    def unravel(idx: torch.Tensor):
        kz = idx // (nY * nX)
        ky = (idx % (nY * nX)) // nX
        kx = idx % nX
        return kz.item(), ky.item(), kx.item()

    # collect the N slices
    slices = []
    for idx in list(lows) + list(highs):
        kz, ky, kx = unravel(idx)

        z0, y0, x0 = kz * stride[0], ky * stride[1], kx * stride[2]
        z1, y1, x1 = z0 + patch_dims[0], y0 + patch_dims[1], x0 + patch_dims[2]

        z_mid = (z0 + z1) // 2                         # axial mid-slice
        cubelet_slice = ct_vol[z_mid, y0:y1, x0:x1]    # (H,W)
        slices.append(cubelet_slice)

    fig, axs = plt.subplots(2, num_cubelets, figsize=(3*num_cubelets, 6))

    # normalise CT voxel values for display
    vmin, vmax = np.percentile(ct_vol, (1, 99))

    for i, sl in enumerate(slices[:num_cubelets]):      # lowest
        axs[0, i].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f"loc={loc[lows[i]].item():.1f}")
    for i, sl in enumerate(slices[num_cubelets:]):      # highest
        axs[1, i].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f"loc={loc[highs[i]].item():.1f}")

    for ax in axs.flat:
        ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[green][✓] Saved best/worst cubelet grid to: [/green]{save_path}")