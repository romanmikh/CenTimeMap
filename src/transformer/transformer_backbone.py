# pylint: disable=all
# type: ignore

import torch
from torch import nn
from rich import print
from einops import pack, rearrange
from vector_quantize_pytorch import VectorQuantize

from src.utils.settings import *
from src.transformer.attention import (
    ContinuousPositionBias,
    Transformer,
    Attention
)
from src.transformer.patch_embed import (
    NoOverlapPatchEmbed,
    OverlapPatchEmbed,
    Conv3dPatchEmbed,
    BinarizedPatchIndicator,
)


# helpers
def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def triplet(val):
    ret = (val, val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 3
    return ret


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        depth,
        dim_head=DIM_HEAD,
        heads=NUM_HEADS,
        channels=CHANNELS,
        attn_dropout=ATTN_DROPOUT,
        ff_dropout=FF_DROPOUT,
    ):
        super().__init__()
        self.image_size = triplet(image_size)
        self.patch_size = triplet(patch_size)
        patch_height, patch_width, patch_depth = self.patch_size
        image_height, image_width, image_depth = self.image_size

        assert (image_height % patch_height) == 0 and \
                (image_width % patch_width)  == 0 and \
                (image_depth % patch_depth)  == 0, (
            f"image size {self.image_size} must be divisible by patch size {self.patch_size}"
        )
        # self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim,
        #                                                    heads=heads,
        #                                                    num_dims=3)
        embedder = {
            "no_overlap": NoOverlapPatchEmbed,
            "unfold_overlap": OverlapPatchEmbed,
            "conv3d_overlap": Conv3dPatchEmbed,
        }.get(EMBED_METHOD)

        if embedder is None:
            raise ValueError(
                f"Unknown embedding method {EMBED_METHOD}. "
                "Available methods: no_overlap, unfold_overlap, conv3d_overlap."
            )
        
        self.to_patch_emb = embedder(channels, dim)
        
        if EMBED_METHOD in ["conv3d_overlap", "unfold_overlap"]:
            self.to_patch_mask = BinarizedPatchIndicator(channels, dim)
        else:
            raise NotImplementedError(
                f"Patch mask for {EMBED_METHOD} is not implemented."
            )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        self.enc_spatial_transformer = Transformer(
            depth=depth, **transformer_kwargs
        )

        # extract attention map deltas for transformer visualisation
        self._deltas = []                          # reset for next call
        def _save_delta(_, __, out):
            self._deltas.append(out)

        for m in self.enc_spatial_transformer.modules():
            if isinstance(m, Attention):
                m.register_forward_hook(_save_delta)

        self.vq = VectorQuantize(
            dim=dim, codebook_size=codebook_size, use_cosine_sim=True
        )

    def encode(self, tokens, token_mask=None):
        """
        Einstein notation:

        b = batch
        d = feature dimension
        x, y, z = image patch sizes
        """
        self._deltas.clear()
        b, z, y, x, _ = tokens.shape
        scan_shape    = (b, z, y, x)
        tokens = rearrange(tokens, "b z y x d -> b (z y x) d")          # (B, N, D)  N = X*Y*Z
        if token_mask is not None:
            token_mask = rearrange(token_mask, "b z y x d -> b (z y x) d")

        # encode with 3D attn_bias
        # attn_bias = self.spatial_rel_pos_bias(z, y, x, device=tokens.device)   # (NUM_HEADS, N, N)
        tokens    = self.enc_spatial_transformer(tokens, attn_bias=None, scan_shape=scan_shape, token_mask=token_mask)
        tokens    = rearrange(tokens, "b (z y x) d -> b z y x d", z=z, y=y, x=x)
        
        # attention deltas 
        if not self._deltas:
            raise RuntimeError("No attention delta hooks stored")

        # (L, B, N, D) – keep every block separate
        layer_deltas = torch.stack(self._deltas, dim=0)          # (L, B, N, D)

        layer_norms   = layer_deltas.norm(dim=-1)                # (L, B, N)  L2-norm over D
        delta_energy  = layer_norms.mean(dim=0)                  # (B, N)     average over layers
        deltas        = delta_energy.view(b, z, y, x)            # (B, Z', Y', X')

        self._deltas.clear() 

        return tokens, deltas

    def forward(self, scan, mask=None):  
        
        assert scan.ndim == 5, "scan must be in the format of (b, d, z, y, x)"
        image_dims = scan.shape[-3:]
        assert tuple(image_dims) == self.image_size

        # print(f"   scan: min {scan.min().item():.1f}, max {scan.max().item():.1f}, mean {scan.mean().item():.1f}, std {scan.std().item():.1f}, median {scan.median().item():.1f}, frac_neg {((scan <= 0).sum().item() / scan.numel()):.3f}, scan sum: {scan.sum():.3f}")
        tokens      = self.to_patch_emb(scan)
        # print(f"tkn_emb: min {tokens.min().item():.1f}, max {tokens.max().item():.1f}, mean {tokens.mean().item():.1f}, std {tokens.std().item():.1f}, median {tokens.median().item():.1f}, frac_neg {((tokens <= 0).sum().item() / tokens.numel()):.3f}, tokens sum: {tokens.sum():.3f}")

        *_, y, x, _ = tokens.shape
        token_mask = self.to_patch_mask(mask) if mask is not None else None
        tokens, deltas = self.encode(tokens, token_mask)
        tokens, _      = pack([tokens], "b * d")
        tokens         = rearrange(tokens, "b (z y x) d -> b z y x d", y=y, x=x)

        # print(f"tknmask: min {tokens.min().item():.1f}, max {tokens.max().item():.1f}, mean {tokens.mean().item():.1f}, std {tokens.std().item():.1f}, median {tokens.median().item():.1f}, frac_neg {((tokens <= 0).sum().item() / tokens.numel()):.3f}, tnmask sum: {tokens.sum():.3f}")
        tokens = tokens * token_mask if token_mask is not None else tokens
        # print(f" tokens: min {tokens.min().item():.1f}, max {tokens.max().item():.1f}, mean {tokens.mean().item():.1f}, std {tokens.std().item():.1f}, median {tokens.median().item():.1f}, frac_neg {((tokens <= 0).sum().item() / tokens.numel()):.3f}, tokens sum: {tokens.sum():.3f}")

        return tokens, deltas