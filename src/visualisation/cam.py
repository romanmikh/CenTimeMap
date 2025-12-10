from typing import Tuple

import numpy as np
from pytorch_grad_cam import (
    KPCA_CAM,
    AblationCAM,
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor, nn

from src.visualisation.base_interp import BaseInterp

cam_methods_map = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "layercam": LayerCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "kpcacam": KPCA_CAM,
}


class CamInterp(BaseInterp):
    """GradCAM Interpreter"""

    def __init__(self, cam_interp_method: str, **kwargs):
        super().__init__(**kwargs)
        assert (
            cam_interp_method in cam_methods_map
        ), f"Invalid CAM method. Choose from {list(cam_methods_map.keys())}"
        self.cam_interp_method = cam_methods_map[cam_interp_method]

    def interpret(self, x: Tensor) -> Tuple[np.ndarray, Tensor]:
        """Interpret the input image using GradCAM"""
        target_layer = self.get_target_layer()
        target = self.model(x).argmax(dim=1)
        target = [ClassifierOutputTarget(target)]

        if isinstance(target_layer, nn.LayerNorm):
            # transformers need an additional argument "reshape_transform"
            def vit_reshape_transform(tensor):
                height = width = np.sqrt(tensor.size(1) - 1).astype(int)
                result = tensor[:, 1:, :].reshape(
                    tensor.size(0), height, width, tensor.size(2)
                )

                # Bring the channels to the first dimension,
                # like in CNNs.
                result = result.transpose(2, 3).transpose(1, 2)
                return result

            with self.cam_interp_method(
                model=self.model,
                target_layers=[target_layer],
                reshape_transform=vit_reshape_transform,
            ) as cam:
                grayscale_cam = cam(input_tensor=x, targets=target)  # type: ignore

        else:
            with self.cam_interp_method(
                model=self.model, target_layers=[target_layer]
            ) as cam:
                grayscale_cam = cam(input_tensor=x, targets=target)  # type: ignore

        pred = cam.outputs
        return grayscale_cam, pred
