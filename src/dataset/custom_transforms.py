"""
Custom transforms for the survival analysis experiments.

This module Implements the custom transforms for the survival analysis experiments.
These transforms are applied to the images and clinical data in the dataset class.

Author: Ahmed H. Shahin
Date: 31/8/2023
"""
# pylint: disable=no-member

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import rotate, zoom
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

from src.dataset.imputation.train_imputation_model import EM, Discretizer
import math
import torch.nn.functional as F


class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.

    Args:
        sample (Dict): A dictionary containing the sample data.
        The values should be either numpy arrays or strings.

    Returns:
        Dict: Updated sample with PyTorch tensors instead of numpy arrays.
    """

    def __call__(
        self, sample: Dict[str, Union[np.ndarray, str]]
    ) -> Dict[str, Union[torch.Tensor, str]]:
        for key, value in sample.items():
            if self._is_string(value) or (key == "clinical_data"):
                continue
            sample[key] = self._convert_to_tensor(value)
        return sample

    @staticmethod
    def _is_string(value: Union[np.ndarray, str]) -> bool:
        return isinstance(value, str)

    @staticmethod
    def _convert_to_tensor(value: np.ndarray) -> torch.Tensor:
        if value.ndim == 3:
            return torch.from_numpy(value[None]).type(torch.FloatTensor)
        return torch.from_numpy(value).type(torch.FloatTensor)


class ToDevice:
    def __init__(self, device, keys):
        self.device = device
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key].to(self.device)#, non_blocking=True)
        return sample

class ImputeMissingData:
    """
    Impute missing data in the clinical data using a latent variable model,
    proposed in https://arxiv.org/abs/2203.11391.

    Args:
        split (str): Data split type ('train', 'val', or 'test'). If test or val,
        we take the argmax of the posterior, otherwise we sample from it during
        training.
        params_path (str): Path to the saved imputation model parameters.
        use_missing_indicator (bool): Whether to use missing indicator.
    """

    def __init__(
        self, split: str, params_path: str, use_missing_indicator: bool = False
    ):
        self.split = split
        self.use_missing_indicator = use_missing_indicator
        self._initialize_model(params_path)

    def _initialize_model(self, params_path: str):
        model_data = np.load(params_path, allow_pickle=True)
        
        discretizer = Discretizer(
            model_data["discretizer_cont_feats_idx"].tolist(), model_data["discretizer_nbins"].tolist()
        )
        discretizer.bins = model_data["discretizer_bins"].tolist()
        discretizer.representative_values = model_data[
            "discretizer_representative_values"
        ]

        self.model = EM(
            num_latent_states=model_data["H"],
            n_iter=0,
            num_categories=model_data["num_categories"],
            discretizer=discretizer,
            num_features=6,
        )
        self.model.p_h = model_data["p_h"]
        self.model.p_x_given_h = model_data["p_x_given_h"].tolist()

    def __call__(self, sample: Dict) -> Dict:
        clinical_data = sample["clinical_data"].copy()
        if self.use_missing_indicator:
            clinical_data, missing_indicator = self._handle_missing_data(clinical_data)

        clinical_data = self._impute_missing_data(clinical_data)

        if self.use_missing_indicator:
            clinical_data = np.concatenate([clinical_data, missing_indicator])

        sample["clinical_data"] = clinical_data
        return sample

    def _handle_missing_data(
        self, clinical_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Reorder to align with the imputation model. 3 cont., disc.: 2 classes,
        # 3 classes, 2 classes
        missing_indicator = np.zeros(3 + 2 + 3 + 2)
        idx_dict = {0: [0], 1: [1], 2: [2], 3: [3, 4], 4: [5, 6, 7], 5: [8, 9]}
        missing_idxs = np.where(np.isnan(clinical_data))[0]
        for missing_idx in missing_idxs:
            missing_indicator[idx_dict[missing_idx]] = 1
        return clinical_data, missing_indicator

    def _impute_missing_data(self, clinical_data: np.ndarray) -> np.ndarray:
        reordered_data = clinical_data.reshape(1, -1)
        # reordered_data = clinical_data[[0, 3, 4, 1, 2, 5]].reshape(1, -1)
        discrete_data = self.model.discretizer.transform(reordered_data)
        imputed_data = self.model.predict(
            discrete_data,
            ("sample" if self.split == "train" else "mean"),
        )
        # Back to continuous
        continuous_data = self.model.discretizer.inverse_transform(imputed_data)
        reordered_data[np.isnan(reordered_data)] = continuous_data[
            np.isnan(reordered_data)
        ]
        # Reorder back to original order
        return reordered_data.reshape(-1)


class NormalizeClinicalData:
    """
    Normalize the clinical data in the sample dictionary.

    Args:
        normalizer (StandardScaler): Scikit-learn StandardScaler object for normalizing
          continuous features.
        encoder (OneHotEncoder): Scikit-learn OneHotEncoder object for encoding
          categorical features.

    Returns:
        Dict: Updated sample with normalized clinical data.
    """

    def __init__(self, normalizer: StandardScaler, encoder: OneHotEncoder):
        self.normalizer = normalizer
        self.encoder = encoder

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        clinical_data = sample["clinical_data"].copy()
        clinical_data = clinical_data.reshape(1, -1)

        continuous_data = clinical_data[:, [0, 1, 2]]
        continuous_data = self.normalizer.transform(continuous_data)

        categorical_data = clinical_data[:, [3, 4, 5]]
        categorical_data = self.encoder.transform(categorical_data)

        normalized_data = np.concatenate(
            [continuous_data, categorical_data, clinical_data[:, 6:]], axis=1
        ).reshape(-1)

        sample["clinical_data"] = torch.from_numpy(normalized_data).type(torch.FloatTensor)
        return sample


class RandomRotate:
    """
    Randomly rotate the image in the sample dictionary within a specified angle range.

    Args:
        angle_range (Tuple[int, int]): The range of angles for random rotation.
          Defaults to (-15, 15).
        prob (float): Probability of applying the rotation. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the rotated image.
    """

    def __init__(self, angle_range: Tuple[int, int] = (-10, 10), prob: float = 0.5):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() <= self.prob:
            rotation_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            image_data = sample["img"]
            rotated_image = rotate(
                image_data,
                rotation_angle,
                reshape=False,
                axes=(1, 2),
                cval=image_data.min(),
            )
            sample["img"] = rotated_image
        return sample

class RandomRotateTorch:
    """
    Randomly rotate the image in the sample dictionary within a specified angle range.

    Args:
        angle_range (Tuple[int, int]): The range of angles for random rotation.
          Defaults to (-15, 15).
        prob (float): Probability of applying the rotation. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the rotated image.
    """

    def __init__(self, angle_range: Tuple[int, int] = (-5, 5), prob: float = 0.5):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, image_data) -> torch.Tensor:
        def rotate_volume(data, angle_degrees):
            angle = math.radians(angle_degrees)  # Convert angle to radians

            data = data.float()
            pad_value = data.min()

            assert data.dim() == 4, "Input must be a 4D tensor"
            data = data.unsqueeze(0)
            
            # Prepare a mask with the same size as the data, initially filled with ones
            mask = torch.ones_like(data)
            # set the borders to zero
            mask[0, 0, :2, :, :] = 0
            mask[0, 0, -1:, :, :] = 0
            mask[0, 0, :, :2, :] = 0
            mask[0, 0, :, -1:, :] = 0
            mask[0, 0, :, :, :2] = 0
            mask[0, 0, :, :, -1:] = 0

            # Create rotation matrix for XY plane
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)

            # This rotation matrix rotates points in the XY plane (around the Z axis)
            rotation_matrix = torch.tensor([
                [cos_val, -sin_val, 0, 0],  # Cos and -Sin for rotation in the Y dimension (Height)
                [sin_val, cos_val, 0, 0],  # Sin and Cos for rotation in the X dimension (Width)
                [0, 0, 1, 0],
                [0, 0, 0, 1]  # Homogeneous coordinate
            ], dtype=torch.float32).to(data.device)

            # Generate grid using only the top 3x4 submatrix
            rotation_matrix = rotation_matrix[:3, :].unsqueeze(0)

            # Generate grid for both data and mask
            grid = F.affine_grid(rotation_matrix, data.size(), align_corners=True)

            # Apply grid sample with zero padding to both data and mask
            rotated_data = F.grid_sample(data, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            rotated_mask = F.grid_sample(mask, grid, mode='nearest', padding_mode='zeros', align_corners=True)

            # Use the mask to apply the pad value where padding occurred
            rotated_data[rotated_mask == 0] = pad_value
            del rotated_mask, mask

            return rotated_data.squeeze(0)

        def rotate_batch(data, angle_degrees):
            pad_value = torch.amin(data, dim=(1,2,3,4))
            pad_value = pad_value.view(-1, 1, 1, 1, 1).expand_as(data)

            angle = torch.deg2rad(angle_degrees)  # Convert angle to radians

            # Create rotation matrix around the specified axes
            cos_vals = torch.cos(angle).view(-1, 1)
            sin_vals = torch.sin(angle).view(-1, 1)
            
            zeros = torch.zeros_like(cos_vals)
            ones = torch.ones_like(cos_vals)

            # Rotation matrices for each batch element
            rotation_matrices = torch.stack([
                torch.cat([cos_vals, -sin_vals, zeros, zeros], dim=1),
                torch.cat([sin_vals, cos_vals, zeros, zeros], dim=1),
                torch.cat([zeros, zeros, ones, zeros], dim=1),
                torch.cat([zeros, zeros, zeros, ones], dim=1)
            ], dim=1)[:, :3]  # Use only the top 3 rows for 3D affine transformation

            rotation_matrices = rotation_matrices.to(data.device)

            # Prepare masks with same size as the data, initially filled with ones
            masks = torch.ones_like(data)
            # Set the borders to zero in the masks
            masks[..., :2, :, :] = 0
            masks[..., -2:, :, :] = 0
            masks[..., :, :2, :] = 0
            masks[..., :, -2:, :] = 0
            masks[..., :, :, :2] = 0
            masks[..., :, :, -2:] = 0

            # Generate grids for each item in the batch using their respective rotation matrices
            grids = F.affine_grid(rotation_matrices, data.size(), align_corners=True)

            # Apply grid sample with zero padding to both data and masks
            rotated_data = F.grid_sample(data, grids, mode='bilinear', padding_mode='zeros', align_corners=True)
            rotated_masks = F.grid_sample(masks, grids, mode='nearest', padding_mode='zeros', align_corners=True)

            # Use the masks to apply the pad value where padding occurred
            rotated_data[rotated_masks == 0] = pad_value[rotated_masks == 0]

            del rotated_masks, masks, grids

            return rotated_data

        # if np.random.uniform() <= self.prob:
        #     rotation_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        #     image_data = sample["img"]
        #     rotated_image = rotate_volume(image_data, rotation_angle)
        #     sample["img"] = rotated_image
        # now this works with a batch of 3D images
        n = image_data.size(0)
        prob_mask = torch.rand(n) <= self.prob
        rotation_angles = torch.FloatTensor(n).uniform_(*self.angle_range)
        rotation_angles = rotation_angles * prob_mask
        if torch.sum(prob_mask) == 0:
            return image_data
        rotated_images = rotate_batch(image_data, rotation_angles)
        return rotated_images

class Windowing:
    """
    Apply windowing to the image in the sample dictionary and normalize to -1 to 1.
    This is specific to CT RATE model.

    Args:
        window_min (int): Minimum value for windowing. Defaults to -1000.
        window_max (int): Maximum value for windowing. Defaults to 200.

    Returns:
        Dict: Updated sample with the windowed image.
    """
    def __init__(self, window_min: int = -1000, window_max: int = 200):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_data = sample["img"]
        if isinstance(image_data, torch.Tensor):
            assert image_data.dim() == 4, "Input must be a 4D tensor"
            assert image_data.size(0) == 1, "Input must have a batch size of 1"
            windowed_image = torch.clamp(image_data, self.window_min, self.window_max)
            windowed_image = torch.transpose(windowed_image, 2, 3)
            windowed_image = torch.flip(windowed_image, [1])
            windowed_image = (windowed_image + 400) / 600
        else:
            windowed_image = np.clip(image_data, self.window_min, self.window_max)
            windowed_image = np.transpose(windowed_image, (0, 2, 1))
            windowed_image = np.flip(windowed_image, axis=0)
            windowed_image = (windowed_image + 400) / 600
        sample["img"] = windowed_image
        return sample


class RandomTranslate:
    """
    Randomly translate the image in the sample dictionary within a specified shift
      range.

    Args:
        shift_range (Tuple[int, int]): The range of shifts for random translation.
          Defaults to (-20, 20).
        prob (float): Probability of applying the translation. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the translated image.
    """

    def __init__(self, shift_range: Tuple[int, int] = (-30, 30), prob: float = 0.5):
        self.shift_range = shift_range
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() <= self.prob:
            shift_z = np.random.uniform(self.shift_range[0], self.shift_range[1])
            shift_y = np.random.uniform(self.shift_range[0], self.shift_range[1])
            shift_x = np.random.uniform(self.shift_range[0], self.shift_range[1])
            image_data = sample["img"]
            translated_image = np.roll(image_data, int(shift_z), axis=0)
            translated_image = np.roll(translated_image, int(shift_y), axis=1)
            translated_image = np.roll(translated_image, int(shift_x), axis=2)
            sample["img"] = translated_image
        return sample

class RandomTranslateTorch:
    """
    Randomly translate the image in the sample dictionary within a specified shift
      range.

    Args:
        shift_range (Tuple[int, int]): The range of shifts for random translation.
          Defaults to (-20, 20).
        prob (float): Probability of applying the translation. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the translated image.
    """

    def __init__(self, shift_range: Tuple[int, int] = (-30, 30), prob: float = 0.5):
        self.shift_range = shift_range
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() <= self.prob:
            shift_z = np.random.uniform(self.shift_range[0], self.shift_range[1])
            shift_y = np.random.uniform(self.shift_range[0], self.shift_range[1])
            shift_x = np.random.uniform(self.shift_range[0], self.shift_range[1])
            shift_z = int(shift_z)
            shift_y = int(shift_y)
            shift_x = int(shift_x)
            image_data = sample["img"]
            assert image_data.dim() == 4, "Input must be a 4D tensor"
            pad_value = image_data.min()
            translated_image = torch.roll(image_data, shift_z, dims=1)
            translated_image = torch.roll(translated_image, shift_y, dims=2)
            translated_image = torch.roll(translated_image, shift_x, dims=3)
            # to avoid merroring
            if shift_z > 0:
                translated_image[0, :shift_z, :, :] = pad_value
            elif shift_z < 0:
                translated_image[0, shift_z:, :, :] = pad_value
            if shift_y > 0:
                translated_image[0, :, :shift_y, :] = pad_value
            elif shift_y < 0:
                translated_image[0, :, shift_y:, :] = pad_value
            if shift_x > 0:
                translated_image[0, :, :, :shift_x] = pad_value
            elif shift_x < 0:
                translated_image[0, :, :, shift_x:] = pad_value
            sample["img"] = translated_image
        return sample


class RandomScale:
    """
    Randomly scale the image in the sample dictionary within a specified scale range.

    Args:
        scale_range (Tuple[float, float]): The range of scales for random scaling.
          Defaults to (0.8, 1.2).
        prob (float): Probability of applying the scaling. Defaults to 0.5.

    Returns:
        Dict: Updated sample with the scaled image.
    """

    def __init__(
        self, scale_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5
    ):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() <= self.prob:
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            image_data = sample["img"]
            scaled_image = zoom(
                image_data, scale_factor, order=1, cval=image_data.min()
            )

            # Resize to original size
            if scale_factor > 1:
                scaled_image = scaled_image[
                    scaled_image.shape[0] // 2
                    - image_data.shape[0] // 2 : scaled_image.shape[0] // 2
                    + image_data.shape[0] // 2,
                    scaled_image.shape[1] // 2
                    - image_data.shape[1] // 2 : scaled_image.shape[1] // 2
                    + image_data.shape[1] // 2,
                    scaled_image.shape[2] // 2
                    - image_data.shape[2] // 2 : scaled_image.shape[2] // 2
                    + image_data.shape[2] // 2,
                ]
            else:
                padding_amount = (image_data.shape[0] - scaled_image.shape[0]) // 2
                padding = (
                    (padding_amount, padding_amount + (scaled_image.shape[0] % 2)),
                    (padding_amount, padding_amount + (scaled_image.shape[1] % 2)),
                    (padding_amount, padding_amount + (scaled_image.shape[2] % 2)),
                )
                scaled_image = np.pad(
                    scaled_image,
                    padding,
                    mode="constant",
                    constant_values=scaled_image.min(),
                )

            sample["img"] = scaled_image
        return sample
