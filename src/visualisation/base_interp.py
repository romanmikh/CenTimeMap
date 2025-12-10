"""
Base class for interpretability methods. Given a trained model and an input image, the
interpretability method should return a heatmap of the importnat regions in the image.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from torch import Tensor, nn


class BaseInterp(ABC):
    """
    Base class for interpretability methods. Given a trained model and an input image, the
    interpretability method should return a heatmap of the importnat regions in the image.

    Args:
        model (nn.Module): The trained model to interpret.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()

    @abstractmethod
    def interpret(self, x: Tensor) -> Tuple[np.ndarray, Tensor]:
        """
        Given an input image, return a heatmap of the important regions in the image and
        the model output for the image.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tuple[np.ndarray, Tensor]: Heatmap of important regions in the image and the
            model output for the image.
        """
        raise NotImplementedError

    def get_target_layer(self):
        """
        Get the target layer for the interpretability method. This layer is the last
        convolutional layer in the model for a CNN. For a transformer model, it is the
        last layer before the classification/regression head.
        """
        target_layer = None
        target_name = None
        for layer_name, layer in list(self.model.named_modules())[:-1]:
            if layer_name == "conv_proj":
                continue
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                target_layer = layer
                target_name = layer_name

        if target_layer is None:
            # transformers
            for layer_name, layer in list(self.model.named_modules())[:-1]:
                if layer_name == "conv_proj":
                    continue
                if isinstance(layer, nn.LayerNorm):
                    target_layer = layer
                    target_name = layer_name

        assert target_layer is not None, "Could not find target layer"
        print(f"Using target layer: {target_name}")
        return target_layer
