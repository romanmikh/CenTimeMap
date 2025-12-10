# src/utils/perturbation.py

import torch
import torch.nn.functional as F
import random

def occlude_patch(volume, center, patch_size, occlusion_value=0.0):
    """
    Sets a cube patch in the 3D volume to a specified occlusion value.

    Args:
        volume (torch.Tensor): Shape (1, D, H, W)
        center (tuple): (x, y, z) center of patch
        patch_size (tuple): (dx, dy, dz)
        occlusion_value (float): Value to occlude the patch with
    """
    occluded = volume.clone()
    x, y, z = center
    dx, dy, dz = patch_size
    _, D, H, W = volume.shape

    x_min = max(x - dx // 2, 0)
    x_max = min(x + dx // 2, D)
    y_min = max(y - dy // 2, 0)
    y_max = min(y + dy // 2, H)
    z_min = max(z - dz // 2, 0)
    z_max = min(z + dz // 2, W)

    occluded[0, x_min:x_max, y_min:y_max, z_min:z_max] = occlusion_value
    return occluded


def compute_perturbation_saliency(
    model,
    input_volume,
    patch_size=(8, 8, 8),
    stride=8,
    class_index=0,
    device="cpu",
    normalize=True,
    use_soft_occlusion=False,
    jitter_radius=1,
    num_jitter_samples=3 # number of jitter samples per patch
):
    """
    Computes perturbation-based saliency map by occluding patches.

    Args:
        model: Trained model
        input_volume (torch.Tensor): Shape (1, D, H, W)
        patch_size (tuple): Size of occlusion cube
        stride (int): Sliding step for occlusion
        class_index (int): Output class to score
        device (str): 'cpu' or 'cuda'
        normalize (bool): Normalize final saliency map
        use_soft_occlusion (bool): Use mean voxel value instead of 0

    Returns:
        saliency (torch.Tensor): Shape (D, H, W)
    """
    model.eval()
    input_volume = input_volume.to(device)
    saliency = torch.zeros_like(input_volume)

    with torch.no_grad():
        base_pred = model(input_volume.unsqueeze(0))  # (1, tmax)
        base_score = base_pred[0, class_index].item()
        _, D, H, W = input_volume.shape

        for x in range(patch_size[0] // 2, D - patch_size[0] // 2, stride):
            for y in range(patch_size[1] // 2, H - patch_size[1] // 2, stride):
                for z in range(patch_size[2] // 2, W - patch_size[2] // 2, stride):
                    
                    score_drops = []

                    for _ in range(num_jitter_samples):
                        jittered_x = min(max(x + random.randint(-jitter_radius, jitter_radius), 0), D - 1)
                        jittered_y = min(max(y + random.randint(-jitter_radius, jitter_radius), 0), H - 1)
                        jittered_z = min(max(z + random.randint(-jitter_radius, jitter_radius), 0), W - 1)

                        occlusion_value = input_volume.mean().item() if use_soft_occlusion else 0.0
                        perturbed = occlude_patch(input_volume, (jittered_x, jittered_y, jittered_z), patch_size, occlusion_value)
                        pred = model(perturbed.unsqueeze(0))
                        score = pred[0, class_index].item()
                        score_drops.append(base_score - score)
                        
                    avg_drop = sum(score_drops) / len(score_drops)
                    saliency[0, x, y, z] = avg_drop  # Assign to center (not jittered position)

        saliency = F.relu(saliency)

        if normalize:
            saliency /= (saliency.max() + 1e-8)

    return saliency.squeeze(0).cpu()
