from torch import nn
from einops.layers.torch import Rearrange
from src.utils.settings import *


class Conv3dPatchEmbed(nn.Module):
    """
    Extract (possibly overlapping) DxHxW cubelets with Conv3d.
    Introduces learnable weights: dim * (channels * pd * ph * pw) (no biases)

    Should be superior to mechanical, parameter-free embed methods:
    1. Conv3d is faster and more memory-efficient than unfold+Linear (dim << patch_volume)
    2. Conv3d preserves spatial information naturally
    3. InstanceNorm adds invariance to brightness and scale
    4. LayerNorm adds stability to training
    5. Each learned filter encodes edges/textures/patterns instead of hard voxels
    6. Reduces noise sensitivity

    Possible issues:
    1. Shared Conv3d weights may leak information outside of lungmask
    2. introduces parameters, may overfit / need regularization
    """
    def __init__(self, channels: int, dim: int):
        super().__init__()
        self.patch_dims = (PATCH_D, PATCH_H, PATCH_W)
        self.stride_3d  = tuple(p // PATCH_OVERLAP for p in self.patch_dims)
        self.embed      = nn.Sequential(
            nn.Conv3d(
                channels,
                dim,
                kernel_size=self.patch_dims,
                stride=self.stride_3d,
                bias=False,
            ),
            nn.InstanceNorm3d(dim, affine=False),        # affine=False removes absolute brightness
            Rearrange("b d z y x -> b z y x d"),
            nn.LayerNorm(dim, elementwise_affine=True)
        )

    def forward(self, x):
        return self.embed(x)        # x : (B, C, Z, Y, X) -> (B, Z', Y', X', D)


class NoOverlapPatchEmbed(nn.Module):
    """
    Extract non-overlapping DxHxW cubelets with Rearrange & Linear.
    Introduces learnable weights from Linear: dim * (channels * pd * ph * pw)
    """
    def __init__(self, channels, dim):
        super().__init__()
        self.embed = nn.Sequential(
            Rearrange(
                "b c (z pd) (y p1) (x p2) -> b z y x (c pd p1 p2)",
                p1=PATCH_H,
                p2=PATCH_W,
                pd=PATCH_D,
            ),
            nn.LayerNorm(channels * PATCH_W * PATCH_H * PATCH_D),
            nn.Linear(channels * PATCH_W * PATCH_H * PATCH_D, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x):
        return self.embed(x)        # x : (B, C, Z, Y, X) -> (B, Z', Y', X', D)


class OverlapPatchEmbed(nn.Module):
    """
    #TODO: bebug. Run to visually see the asymmetry.
    Extract (possibly overlapping) DxHxW cubelets with unfold & Linear.
    Introduces learnable weights: dim * (channels * pd * ph * pw) + dim biases
    """
    def __init__(self, channels, dim):
        super().__init__()
        self.patch_dims = (PATCH_D, PATCH_H, PATCH_W)
        self.voxels     = channels * self.patch_dims[0] * self.patch_dims[1] * self.patch_dims[2]
        self.stride_3d  = tuple(p // PATCH_OVERLAP for p in self.patch_dims)
        # self.embed      = nn.Linear(self.voxels, dim, bias=True)
        self.embed      = nn.Sequential(
            nn.LayerNorm(self.voxels, elementwise_affine=False),
            nn.Linear(self.voxels, dim, bias=False),
            nn.LayerNorm(dim, elementwise_affine=True),
        )

    def forward(self, x):
        # each slide creates a new dimension
        x = (x.unfold(2, self.patch_dims[0], self.stride_3d[0])     # slides along Z axis by pd
               .unfold(3, self.patch_dims[1], self.stride_3d[1])    # slides along Y axis by ph
               .unfold(4, self.patch_dims[2], self.stride_3d[2])    # slides along X axis by pw
               .contiguous())                                       # needed for .view()

        B, C, Dn, Hn, Wn, pd, ph, pw = x.shape
        x = x.view(B, C, Dn, Hn, Wn, -1)                            # flatten last 3 dims
        x = x.permute(0, 2, 3, 4, 1, 5)
        x = x.reshape(B, Dn, Hn, Wn, -1)

        return self.embed(x)        # x : (B, C, Z, Y, X) -> (B, Z', Y', X', D)
    
    
class BinarizedPatchIndicator(nn.Module):
    """
    Extract overlapping 3D cubelets and binarize them.
    Output: (B, Z', Y', X', 1), where each cubelet is 1 if it contains any 1 in the original volume.
    """
    def __init__(self, channels: int, dim: int):
        super().__init__()
        self.patch_dims = (PATCH_D, PATCH_H, PATCH_W)
        self.stride_3d  = tuple(p // PATCH_OVERLAP for p in self.patch_dims)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, Z, Y, X), binary values {0, 1}
        Returns:
            patches: Tensor of shape (B, Z', Y', X', 1), binary values
        """
        pd, ph, pw = self.patch_dims
        sd, sh, sw = self.stride_3d

        # unfold and extract cubelets
        x = (x.unfold(2, pd, sd)
               .unfold(3, ph, sh)
               .unfold(4, pw, sw))  # (B, C, Z', Y', X', pd, ph, pw)

        # flatten cubelets
        B, C, Zp, Yp, Xp, pd_, ph_, pw_ = x.shape
        x = x.contiguous().view(B, C, Zp, Yp, Xp, -1)  # (B, C, Z', Y', X', patch_volume)

        # binarize: cubelet == 1 if more than 1/D voxels are 1
        patch_volume = pd * ph * pw
        threshold = patch_volume // 3 * 2 + 1
        x_sum = x.sum(dim=-1, keepdim=True)  # sum over flattened patch
        x = (x_sum >= threshold).float()  # (B, C, Z', Y', X', 1)

        # if input had multiple channels, reduce across them too
        x = x.amax(dim=1, keepdim=False)  # (B, Z', Y', X', 1)

        return x