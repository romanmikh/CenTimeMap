import torch
import numpy as np
from torch import nn
from rich import print
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM  
from scipy.ndimage import zoom, gaussian_filter
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
from PIL import Image
import pyvista as pv

from src.utils.settings import *


# -------------------------------------------------------------------------------------
# heatmap getters
# -------------------------------------------------------------------------------------
def get_head_feats_and_heat(model, vol_gpu):
    """Returns backbone feature tensors & raw location heatmap from the head"""
    feats, _ = model.backbone(vol_gpu)                     # (1, Z', Y', X', D)
    _, loc_vec = model.head(feats, return_loc=True)     # _, (1, N)
    Z, Y, X = feats.shape[1:4]
    heat = loc_vec.view(Z, Y, X)                        # (Z', Y', X')
    return feats, heat


def get_transformer_feats_heat_deltas(model, vol_gpu):
    """Return backbone feature tensors, their L2-norm heat-map & attention deltas"""
    feats, deltas = model.backbone(vol_gpu)            # (1,Z',Y',X',D), (B, Z', Y', X')
    feats = feats - feats.mean(dim=-1, keepdim=True)   
    heat  = feats.norm(dim=-1)[0]                      # (Z',Y',X') - L2-norm marginalises D dimension, not optimal
    return feats, heat, deltas[0]                      # (1,Z',Y',X',D), (Z', Y', X'), (Z', Y', X')


def get_grad_heat(model, vol_gpu, target="backbone"):
    """Compute a Grad-CAM heat-map either from a selected CNN layer or from transformer backbone"""
    target = target.lower() if isinstance(target, str) else "backbone"

    # Choose CAM layer and model 
    if target == "cnn":
        layer = get_target_CNN_layer(model)
        cam_model = model
        reshape = None
    elif target in ["transformer", "backbone"]:
        layer = model.backbone.to_patch_emb.embed[0]
        cam_model = model.backbone
        reshape = reshape_transform
    else:
        raise ValueError(f"[ERROR] Unknown target: {target}")

    # print(f"[DEBUG] Using layer: {layer}")
    cam = GradCAM(model=cam_model, target_layers=[layer], reshape_transform=reshape)

    # Define GradCAM target
    def target_fn(output):
        if isinstance(output, torch.Tensor):
            result = output.sum() 
        elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            result = output[0].sum()
        else:
            raise ValueError(f"Unexpected model output: {type(output)}")
        
        if result.ndim != 0:
            raise ValueError(f"target_fn must return a scalar. Got: {result.shape}")
        
        return result

    # Run GradCAM
    vol_gpu.requires_grad = True
    with torch.enable_grad():
        heat_np = cam(input_tensor=vol_gpu, targets=[target_fn])[0]  # numpy (Z',Y',X')
    # print(f"[DEBUG] cam.activations_and_grads.gradients: {cam.activations_and_grads.gradients}")
    return torch.from_numpy(heat_np).to(vol_gpu.device)
    

# -------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------
def save_or_viz_plot(plot, save_path=None):
    """Helper function to save or show a pyvista plot"""
    plot.link_views()
    off_scr = save_path is not None
    plot.off_screen = off_scr
    if save_path is None:
        plot.show()
    else:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"[green][✓] Saved image to: [/green]{save_path}")
        plot.show(screenshot=save_path)


def normalise_ct(ct_vol):
    """Normalise CT volume to [0,1] for visualization"""
    # use either np.percentile or np.nanpercentile depending on whether there are nans
    lo, hi  = np.percentile(ct_vol, (1, 99)) if not np.isnan(ct_vol).any() else np.nanpercentile(ct_vol, (1, 99))
    ct_norm = (np.clip(ct_vol, lo, hi) - lo) / (hi - lo + 1e-6)
    return ct_norm


def normalise_heat(heat):
    """Normalise heatmap to [0,1] for visualization"""
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return heat_norm


def reshape_transform(feats: torch.Tensor) -> torch.Tensor:
    """Convert backbone output  (B, Z', Y', X', D) ->(B, D, Z', Y', X') as expected by Grad-CAM"""
    if feats.ndim == 3:  # Transformer token: (B, N, D)
        B, N, D = feats.shape
        side = round(N ** (1/3))  # assume cubic layout
        return feats.permute(0, 2, 1).reshape(B, D, side, side, side)
    elif feats.ndim == 5:  # Already (B, D, Z, Y, X)
        return feats
    else:
        raise ValueError(f"Unexpected feats shape: {feats.shape}")
    #return torch.einsum('b z y x d -> b d z y x', feats)


def get_target_CNN_layer(model: nn.Module) -> nn.Module:
    """
    Select best layer to apply Grad-CAM t (ideally layer closest to output)
    Fallback: any Conv3d, finally a LayerNorm

    Grad-CAM not great because it just interpolates the last, low resolution layer
    it gets worse with more layers    
    """
    for _, layer in reversed(list(model.named_modules())):
        if isinstance(layer, nn.Conv3d) and layer.groups == 1:
            print(f"Using target layer: {layer}")
            return layer

    for _, layer in reversed(list(model.named_modules())):
        if isinstance(layer, nn.Conv3d):
            print(f"Using target layer: {layer}")
            return layer

    for _, layer in reversed(list(model.named_modules())):
        if isinstance(layer, nn.LayerNorm):
            print(f"Using target layer: {layer}")
            return layer

    raise RuntimeError("No suitable target layer found")


def _scale_overlapped_heatmap(token_heat: np.ndarray,
                          ct_shape: tuple[int, int, int],
                          patch_dims: tuple[int, int, int] = (PATCH_D, PATCH_H, PATCH_W),
                          overlap: int = PATCH_OVERLAP) -> np.ndarray:
    """
    token_heat : (Z', Y', X') array returned by the backbone / Grad-CAM.
    ct_shape   : shape of the original scan, e.g. (64, 64, 64).
    patch_dims : size of a cubelet in voxels.
    overlap    : PATCH_OVERLAP from settings.py.


    returns  : (Z, Y, X) array aligned with ct_vol
    """
    stride = tuple(p // overlap for p in patch_dims)            # (2, 2, 2) if overlap=2
    vol    = np.zeros(ct_shape, dtype=np.float32)
    count  = np.zeros(ct_shape, dtype=np.uint16)

    Zt, Yt, Xt = token_heat.shape
    for kz in range(Zt):
        z0 = kz * stride[0]; z1 = z0 + patch_dims[0]            # (0, 4), (2, 6), ...
        for ky in range(Yt):
            y0 = ky * stride[1]; y1 = y0 + patch_dims[1]
            for kx in range(Xt):
                x0 = kx * stride[2]; x1 = x0 + patch_dims[2]
                vol[z0:z1, y0:y1, x0:x1]  += token_heat[kz, ky, kx]
                count[z0:z1, y0:y1, x0:x1] += 1

    count[count == 0] = 1                                       # avoid divide-by-0
    return vol / count                                          # average overlapping tokens    


def correct_heatmap_scaling(token_heat: np.ndarray, 
                   ct_shape: tuple[int, int, int],
                   viz_grads: bool = False,
                   smooth_sigma: float | None = None) -> np.ndarray:
    """
    Convert token-grid heat to full-resolution
    If patches overlap we scale, otherwise default to trilinear zoom
    """
    if PATCH_OVERLAP > 1 and not viz_grads:
        heat =  _scale_overlapped_heatmap(token_heat, ct_shape)
        if smooth_sigma is None:                                # default: half the stride
            stride = tuple(p // PATCH_OVERLAP for p in (PATCH_D, PATCH_H, PATCH_W))
            smooth_sigma = 0.5 * np.mean(stride)
        return gaussian_filter(heat, sigma=smooth_sigma, mode="nearest")
    
    factors = [ct_shape[i] / token_heat.shape[i] for i in range(3)]
    return zoom(token_heat, zoom=factors, order=1)


# def interp_norm_visualise(heat, ct_slice, save_path=None):
#     heat = torch.nn.functional.interpolate(
#         heat.unsqueeze(0).unsqueeze(0),
#         size=ct_slice.shape,
#         mode="bilinear",
#         align_corners=False)[0, 0].cpu()
#     heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
#     rgb = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-6)
#     rgb = rgb.repeat(3, 1, 1).permute(1, 2, 0).numpy().astype("float32")
    
#     vis = show_cam_on_image(rgb, heat.numpy().astype("float32"), use_rgb=True)
#     plt.imshow(vis); plt.axis("off")
#     if save_path is None:
#         plt.show()
#     else:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, bbox_inches="tight", dpi=300)
#         plt.close()


# def viz_feats_slice(feats, ct_slice, save_path=None):
#     """Visualise a single slice of the CT and the corresponding feature heat-map."""
#     heat3d   = feats.norm(dim=-1)[0].cpu().numpy()                      # (Z',Y',X')
#     # print(f"heat shape: {heat3d.shape}")
#     heat_vol = correct_heatmap_scaling(heat3d, (ct_slice.shape[0],)*3)
#     heat_mid = heat_vol[heat_vol.shape[0] // 2]                         # middle axial slice (Y',X')
#     interp_norm_visualise(torch.from_numpy(heat_mid), ct_slice, save_path)


# def viz_grad_slice(heat_3d, ct_slice, save_path=None):
#     """Overlay Grad-CAM heat-map on ct_slice"""
#     # print(f"heat_3d shape: {heat_3d.shape}")
#     if isinstance(heat_3d, np.ndarray):
#         heat_3d = torch.from_numpy(heat_3d)

#     heat = heat_3d[heat_3d.shape[0] // 2]                               # middle axial slice (Y',X')
#     interp_norm_visualise(heat, ct_slice, save_path)


# -------------------------------------------------------------------------------------
# debugging / testing utils
# -------------------------------------------------------------------------------------
def save_lungmask_mid_slice(mask_gpu, path="lungmask_mid_slice.png"):
    """Save mid-slice of lung mask for quick visual check"""
    z  = mask_gpu.shape[-3] // 2
    sl = mask_gpu[0, 0, z].detach().cpu().numpy()
    Image.fromarray(((sl == 0).astype('uint8') * 255)).save(path)


def viz_lungmask_binary(mask_tensor, save_path=None):
    """Visualise lung mask only in 3D"""
    mask = mask_tensor.detach().cpu().numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor
    vol  = mask.astype(np.uint8)
    off  = save_path is not None
    p    = pv.Plotter(off_screen=off, window_size=(900, 700))
    # vol  = np.repeat(vol, 3, axis=0)
    p.add_volume(vol,
                 cmap="gray_r",              # 1 -> black
                 opacity=[0, 1],             # 0 transparent, 1 opaque
                 shade=False)
    save_or_viz_plot(p, save_path=save_path)






# # -------------------------------------------------------------------------------------
# # perturbation & saliency (Youngeun)
# # -------------------------------------------------------------------------------------
# def compare_saliency_and_gradcam(
#     saliency_path: str = "results/perturbation_saliency_volume.npy",
#     gradcam_path: str = "results/gradcam_overlay.png",
#     ct_path: str = None,
#     save_path: str = "results/saliency_vs_gradcam.png"
# ):
#     """Side-by-side comparison of perturbation saliency and GradCAM"""
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#     # Left: Perturbation Saliency
#     try:
#         sal = np.load(saliency_path)
#         mid = sal.shape[0] // 2
#         axs[0].imshow(sal[mid], cmap="hot")
#         axs[0].set_title("Perturbation Saliency (Mid Slice)")
#     except Exception as e:
#         axs[0].text(0.5, 0.5, f"Error loading saliency:\n{e}", ha='center', va='center')
#         axs[0].set_title("Perturbation Saliency")

#     # Right: GradCAM Overlay
#     try:
#         grad_img = np.array(Image.open(gradcam_path))
#         axs[1].imshow(grad_img)
#         axs[1].set_title("GradCAM Overlay (Mid Slice)")
#     except Exception as e:
#         axs[1].text(0.5, 0.5, f"Error loading GradCAM:\n{e}", ha='center', va='center')
#         axs[1].set_title("GradCAM")

#     for ax in axs:
#         ax.axis("off")

#     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight", dpi=300)
#     plt.close()
#     print(f"[✓] Saved saliency vs GradCAM comparison to: {save_path}")


# # -------------------------------------------------------------------------------------
# # Unified CAM Runner
# # -------------------------------------------------------------------------------------
# from pytorch_grad_cam import GradCAM, HiResCAM

# CAM_METHODS = {
#     "gradcam": GradCAM,
#     "hirescam": HiResCAM,
# }


# def run_cam(
#     model,
#     input_volume,
#     cam_type="gradcam",
#     target_layer=None,
#     class_index=None,
#     device="cuda",
#     save_path="results/cam_overlay.png",
# ):
#     """
#     Run a CAM method on a model and CT scan volume.

#     Args:
#         model: TransformerWithHead model
#         input_volume: (1, 1, Z, Y, X) input tensor
#         cam_type: One of 'gradcam', 'gradcam++', 'ablationcam', 'xgradcam'
#         target_layer: Layer to attach CAM to (defaults to patch embedding)
#         class_index: Optional index of class to target (defaults to argmax)
#         device: 'cuda' or 'cpu'
#         save_path: Path to save overlay image

#     Returns:
#         cam_map: (Z, Y, X) heatmap as numpy array
#     """
#     assert cam_type in CAM_METHODS, f"[ERROR] Unsupported CAM type: {cam_type}"

#     model.eval().to(device)
#     input_volume = input_volume.to(device)
#     input_volume.requires_grad = True

#     # Default to patch embedding if no target layer provided
#     if target_layer is None:
#         target_layer = model.backbone.to_patch_emb.embed[0]
#         reshape_transform = globals()["reshape_transform"]  # use already defined function in visualisation.py
#     else:
#         reshape_transform = None

#     cam_cls = CAM_METHODS[cam_type]
#     cam = cam_cls(model=model.backbone, target_layers=[target_layer], reshape_transform=reshape_transform)

#     def target_fn(output):
#         if isinstance(output, torch.Tensor):
#             if class_index is None:
#                 return output.sum()
#             return output[:, class_index].sum()
#         elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
#             return output[0][:, class_index].sum() if class_index is not None else output[0].sum()
#         raise ValueError(f"Unexpected model output: {type(output)}")

#     with torch.enable_grad():
#         grayscale_cam = cam(input_tensor=input_volume, targets=[target_fn])[0]  # shape (Z, Y, X)

#     # Mid-slice visualization
#     ct_np = input_volume[0, 0].detach().cpu().numpy()  # (Z, Y, X)
#     mid_z = ct_np.shape[0] // 2
#     ct_slice = ct_np[mid_z]
#     cam_slice = grayscale_cam[mid_z]

#     ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-6)
#     cam_norm = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min() + 1e-6)

#     overlay = show_cam_on_image(np.stack([ct_norm]*3, axis=-1), cam_norm.astype("float32"), use_rgb=True)

#     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.imsave(save_path, overlay)
#     print(f"[✓] Saved CAM overlay to: {save_path}")

#     return grayscale_cam

# def compare_ct_and_saliency_grid(
#     save_path="results/saliency_composite_grid.png"
# ):
#     """
#     Plot input CT + GradCAM + HiResCAM + Perturbation in 1x4 grid.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from PIL import Image

#     titles = ["CT Scan", "GradCAM", "HiResCAM", "Perturbation"]
#     file_paths = [
#         None,  # CT from .npy
#         "results/gradcam_overlay.png",
#         "results/hirescam_overlay.png",
#         None  
#     ]

#     ct_vol = np.load("results/ct_volume.npy")        # (Z, Y, X)
#     saliency = np.load("results/perturbation_saliency_volume.npy")  # (Z, Y, X)
#     z_center = ct_vol.shape[0] // 2
#     ct_slice = ct_vol[z_center]
#     saliency_slice = saliency[z_center]

#     fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

#     # Plot CT
#     axs[0].imshow(ct_slice, cmap="gray")
#     axs[0].set_title(titles[0])

#     # Plot GradCAM and HiResCAM
#     for i in [1, 2]:
#         try:
#             img = np.array(Image.open(file_paths[i]))
#             axs[i].imshow(img)
#         except Exception:
#             axs[i].text(0.5, 0.5, "Missing", ha='center', va='center')
#         axs[i].set_title(titles[i])

#     # Plot perturbation overlay directly
#     axs[3].imshow(ct_slice, cmap="gray")
#     im = axs[3].imshow(saliency_slice, cmap="hot", alpha=0.5)
#     axs[3].set_title(titles[3])

#     # Equalize formatting
#     for ax in axs:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_aspect("equal")

#     # Add colorbar for perturbation
#     fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"[✓] Saved composite saliency grid to: {save_path}")