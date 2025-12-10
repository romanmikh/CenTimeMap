from src.utils.train_utils import get_device
import torch
import numpy as np
from torch import nn
import pyvista as pv
from rich import print
from pathlib import Path
from functools import partial
from scipy.ndimage import zoom, gaussian_filter
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
from typing import Literal, Optional, Tuple
from lungmask import LMInferer
from src.visualisation.viz_worst_best_cubelets import viz_worst_best_cubelets

from src.utils.settings import *
from src.utils.train_utils import is_using_dummydataset
from src.visualisation.utils import *

DATA_PATH = "/home/rocky/Downloads/david_talk/scan_h5"
HEATMAP_PATH = "/home/rocky/Downloads/david_talk/heatmap_npy"


def _compute_heat(to_visualise, model, ct_gpu):
    """Return (heat, deltas_or_None, is_gradcam_mode)."""
    if to_visualise == "head_feats":
        with torch.no_grad():
            _, heat = get_head_feats_and_heat(model, ct_gpu)
        return heat, None, False

    elif to_visualise == "transformer_feats":
        with torch.no_grad():
            _, heat, deltas = get_transformer_feats_heat_deltas(model, ct_gpu)
        return heat, deltas, False

    elif to_visualise == "pre_transformer_gradcam":
        heat = get_grad_heat(model, ct_gpu, target="CNN")
        return heat, None, True

    elif to_visualise == "post_transformer_gradcam":
        heat = get_grad_heat(model, ct_gpu, target="backbone")
        return heat, None, True

    else:
        raise ValueError("Invalid to_visualise mode")


def get_visualisation_heat(
    to_visualise,
    model,
    ct_gpu,
    ct_vol_shape: Tuple[int, int, int],
    viz_deltas: bool = False,
) -> "np.ndarray":
    heat, deltas, is_grad = _compute_heat(to_visualise, model, ct_gpu)

    # pick tensor (prefer deltas if requested and available)
    tensor = deltas if (viz_deltas and deltas is not None) else heat
    arr = tensor.detach().float().cpu().numpy()
    return correct_heatmap_scaling(
        arr, ct_vol_shape, viz_grads=is_grad
    )  # TODO fix viz_grads=True bloating post_transformer


def load_static_heatmap_delete_this_funciton():
    """Temporary function to load a static heatmap from disk for testing the overlay viz"""
    heatmap_dir = HEATMAP_PATH
    hm_files = sorted(Path(heatmap_dir).glob("*.npy"))
    if not hm_files:
        print("[yellow]No .npy heatmaps found in HEATMAP_PATH[/yellow]")
        return
    heat = np.squeeze(np.load(hm_files[0]))  # (Z,Y,X) expected
    return heat


# -------------------------------------------------------------------------------------
# visualisers
# -------------------------------------------------------------------------------------
def viz_ct_mask_heat_overlay(
    model,
    loader,
    save_path=None,
    to_visualise="head_feats",  # "head_feats" | "transformer_feats" | "pre_transformer_gradcam" | "post_transformer_gradcam" | "all_methods"
    viz_deltas: bool = False,
    use_lungmask: bool = False,
    device=get_device(),
):
    """
    Renders 4x1 grid:
        - CT
        - lungmasked CT
        - heatmap
        - lungmasked CT + heatmap overlay

    or 4x2 grid:
        as above plus:
        - pre-transformer gradcam
        - post-transformer gradcam
        - transformer features
        - blank for now
    """
    # get CT sample
    model.to(device).eval()
    sample = next(iter(loader))
    ct_gpu = sample["img"][:1].to(device)  # (1,1,Z,Y,X) tensor
    ct_vol = ct_gpu[0, 0].cpu().numpy()  # (Z,Y,X) numpy

    # TODO use actual heatmap instead of hardcoded path
    # heat = load_static_heatmap_delete_this_funciton()

    # get heatmap from model (dummy dataset only) TODO for real dataset
    heat = get_visualisation_heat(
        to_visualise="head_feats",
        model=model,
        ct_gpu=ct_gpu,
        ct_vol_shape=ct_vol.shape,
        viz_deltas=viz_deltas,
    )
    print(f"[blue][i] CT volume min/max: [/blue]{ct_vol.min():.2f}/{ct_vol.max():.2f}")
    print(f"[blue][i] Heatmap min/max:   [/blue]{heat.min():.2f}/{heat.max():.2f}")

    # TODO move lungmask logic outside of visualisation, same for other viz functions
    if use_lungmask and not is_using_dummydataset():
        lung_mask_np = LMInferer(batch_size=1).apply(ct_vol).astype(np.uint8)
        mask = lung_mask_np > 0  # otherwise left lung =1 right lung =2
        heat[~mask] = 0.0  # apply lungmask if real dataset only
        print(
            f"[red][i] Heatmap min/max after mask: [/red]{heat.min():.2f}/{heat.max():.2f}"
        )

    off_scr = save_path is not None
    shape = (2, 4) if to_visualise == "all_methods" else (1, 2)
    window_size = (2400, 1200) if to_visualise == "all_methods" else (1200, 600)
    plot = pv.Plotter(
        shape=shape, border=False, window_size=window_size, off_screen=off_scr
    )
    add_custom_text = partial(
        plot.add_text, position="upper_edge", font_size=12, shadow=True, font="arial"
    )

    # 1) whole CT
    plot.subplot(0, 0)
    idx = 1
    # ct_vol = np.repeat(ct_vol, 3, axis=0)
    plot.add_volume(
        ct_vol,
        cmap="gray",
        opacity="linear",
        scalar_bar_args={"title": " " * idx},
    )
    add_custom_text("CT")

    # 2) lungs only
    # plot.subplot(0, 1); idx += 1
    # lung_vol = ct_vol.detach().cpu().numpy() if isinstance(ct_vol, torch.Tensor) else ct_vol
    # if use_lungmask and not is_using_dummydataset():
    #     lung_vol[~mask] = 0.0 # apply lungmask if real dataset only
    # # lung_vol = np.repeat(lung_vol, 3, axis=0)
    # plot.add_volume(
    #     lung_vol,
    #     cmap="gray",
    #     opacity="linear",
    #     scalar_bar_args={'title': ' '*idx},
    #     )
    # add_custom_text("Masked (Lungs Only)")

    # 3) heatmap
    # heat = np.repeat(heat, 3, axis=0)
    base_heat_opac = pv.plotting.tools.opacity_transfer_function(
        "linear_r", n_colors=256
    )
    plot.subplot(0, 1)
    idx += 1
    plot.add_volume(
        heat,
        cmap="coolwarm",
        opacity="sigmoid_20",
        scalar_bar_args={"title": " " * idx},
    )
    add_custom_text("CenTime Heatmap")

    # 4) overlay
    # plot.subplot(0, 3); idx += 1
    # base_ct_opac = pv.plotting.tools.opacity_transfer_function('linear_r', n_colors=256)
    # plot.add_volume(heat, cmap="ice", opacity=base_heat_opac*0.1,  scalar_bar_args={'title': ' '*idx})
    # plot.add_volume(lung_vol,  cmap="gray", opacity=base_ct_opac*0.02, show_scalar_bar=False)
    # add_custom_text("CT + CenTime Heatmap")

    if to_visualise == "all_methods":

        def plot_manager(
            plot, row, col, to_visualise, cmap, opacity_name, opacity_scale, title, idx
        ):
            heat = get_visualisation_heat(
                to_visualise=to_visualise,
                model=model,
                ct_gpu=ct_gpu,
                ct_vol_shape=ct_vol.shape,
                viz_deltas=viz_deltas,
            )
            plot.subplot(row, col)
            base_heat_opac = pv.plotting.tools.opacity_transfer_function(
                opacity_name, n_colors=256
            )
            plot.add_volume(
                heat,
                cmap=cmap,
                opacity=base_heat_opac * opacity_scale,
                scalar_bar_args={"title": " " * idx},
            )
            add_custom_text(title)

        # 5) pre-transformer gradcam
        plot_manager(
            plot,
            1,
            0,
            "pre_transformer_gradcam",
            "Greens",
            "linear",
            0.5,
            "Pre-Transformer GradCAM",
            5,
        )
        # 6) post-transformer gradcam
        plot_manager(
            plot,
            1,
            1,
            "post_transformer_gradcam",
            "Purples",
            "linear",
            0.5,
            "Post-Transformer GradCAM",
            6,
        )
        # 7) transformer features
        plot_manager(
            plot,
            1,
            2,
            "transformer_feats",
            "Reds",
            "linear_r",
            0.1,
            "Transformer Features",
            7,
        )
        # 8) blank for now TODO add more viz methods (PolyCam, GTmask etc.)
        plot.subplot(1, 3)
        add_custom_text(" ")

    save_or_viz_plot(plot, save_path=save_path)


def viz_dummy_heatmap_debug(
    model,
    loader,
    save_path=None,
    to_visualise="head_feats",  # "head_feats" | "transformer_feats" | "pre_transformer_gradcam" | "post_transformer_gradcam"
    viz_deltas=False,
    device=get_device(),
    cmap="gray",
):
    """3x3 visualisation grid of heatmap with different scalings. No lungmask. For dummy dataset only"""
    # get CT sample
    model.to(device).eval()
    sample = next(iter(loader))
    ct_gpu = sample["img"][:1].to(device)  # (1,1,Z,Y,X) tensor
    ct_vol = ct_gpu[0, 0].cpu().numpy()  # (Z,Y,X) numpy

    # get heatmap from model (dummy dataset only) TODO for real dataset
    heat = get_visualisation_heat(
        to_visualise=to_visualise,
        model=model,
        ct_gpu=ct_gpu,
        ct_vol_shape=ct_vol.shape,
        viz_deltas=viz_deltas,
    )
    print(f"[blue][i] CT volume min/max: [/blue]{ct_vol.min():.2f}/{ct_vol.max():.2f}")
    print(f"[blue][i] Heatmap min/max: [/blue]{heat.min():.2f}/{heat.max():.2f}")

    # plot
    off_scr = save_path is not None
    opacities = [
        "linear_r",
        "linear",
        "linear",
        "linear_r",
        "linear",
        "linear_r",
        "linear",
        "linear_r",
    ]
    flip_scal = [False, False, True, True, False, False, True, True]
    cmapsssss = ["gray", "gray", "gray", "gray", "gray_r", "gray_r", "gray_r", "gray_r"]
    plot = pv.Plotter(
        shape=(3, 3), border=False, window_size=(1600, 1200), off_screen=off_scr
    )
    base_ct_opac = pv.plotting.tools.opacity_transfer_function(
        "linear", n_colors=256
    )  # add _r if brightness is -ve, cmap works both ways. Scale bar is off because "linear" blends

    plot.subplot(0, 0)
    plot.add_volume(ct_vol, cmap="gray", opacity=base_ct_opac * 1)
    plot.add_text("CT", font_size=10)

    for i, (opacity, flip, cmap) in enumerate(zip(opacities, flip_scal, cmapsssss)):
        plot.subplot((i + 1) // 3, (i + 1) % 3)
        base_heat_opac = pv.plotting.tools.opacity_transfer_function(
            opacity, n_colors=256
        )
        plot.add_volume(
            heat,
            cmap=cmap,
            flip_scalars=flip,
            opacity=base_heat_opac * 0.2,
            scalar_bar_args={"title": " " * i},
        )
        plot.add_text(f"{to_visualise} ({opacity}, {flip}, {cmap})", font_size=10)

    save_or_viz_plot(plot, save_path=save_path)


def viz_all(model, val_loader, args):
    """Wrapper to call all visualisation functions at inference"""
    viz_ct_mask_heat_overlay(
        model,
        val_loader,
        save_path=f"results/_presentation_{args.method}_{args.head}.png",
        to_visualise="head_feats",
        use_lungmask=args.use_lungmask,
    )

    # viz_ct_mask_heat_overlay(
    #     model,
    #     val_loader,
    #     save_path=f"results/_presentation_{args.method}_{args.head}_all_methods.png",
    #     to_visualise="all_methods",
    #     use_lungmask=args.use_lungmask,
    # )

    # viz_dummy_heatmap_debug(
    #     model,
    #     val_loader,
    #     save_path=f"results/head_feats_{args.method}_{args.head}.png",
    #     to_visualise="head_feats",
    # )

    # viz_dummy_heatmap_debug(
    #     model,
    #     val_loader,
    #     save_path=f"results/transformer_feats_{args.method}_{args.head}.png",
    #     to_visualise="transformer_feats",
    # )

    # viz_dummy_heatmap_debug(
    #     model,
    #     val_loader,
    #     save_path=f"results/post_transformer_gradcam_{args.method}_{args.head}.png",
    #     to_visualise="post_transformer_gradcam",
    # )

    # viz_dummy_heatmap_debug(
    #     model,
    #     val_loader,
    #     save_path=f"results/pre_transformer_gradcam_{args.method}_{args.head}.png",
    #     to_visualise="pre_transformer_gradcam",
    # )

    # if args.head == "interp":
    #     cubelet_path = f"results/worst_best_cubelets_{args.method}_{args.head}.png"
    #     viz_worst_best_cubelets(model, val_loader, save_path=cubelet_path)
