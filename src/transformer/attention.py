# pylint: disable=all
# type: ignore

import math
from typing import Tuple
from rich import print
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, repeat
from torch import einsum, nn
from src.utils.settings import *

# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def l2norm(t):
    return F.normalize(t, dim=-1)


# helpers for windowed attention
def window_partition(x, window_size):
    """
    Args:
        x: (B, Dx, Dy, Dz, C)
        window_size: int or tuple of 3 ints
    Returns:
        windows: (num_windows*B, wx*wy*wz, C)
    """
    B, Dx, Dy, Dz, C = x.shape
    wx, wy, wz = window_size
    # Pad if needed
    pad_x = (0, (wx - Dx % wx) % wx)
    pad_y = (0, (wy - Dy % wy) % wy)
    pad_z = (0, (wz - Dz % wz) % wz)
    x = F.pad(x, (0, 0, *pad_z, *pad_y, *pad_x))
    Dx_p, Dy_p, Dz_p = x.shape[1:4]
    # Partition windows
    x = x.view(
        B,
        Dx_p // wx, wx,
        Dy_p // wy, wy,
        Dz_p // wz, wz,
        C
    )
    # (B, Dx_blocks, wx, Dy_blocks, wy, Dz_blocks, wz, C)
    windows = x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, wx*wy*wz, C)
    return windows, (Dx_p, Dy_p, Dz_p)


def window_reverse(windows, window_size, original_shape):
    """
    Args:
        windows: (num_windows*B, wx*wy*wz, C)
        window_size: tuple of 3 ints
        original_shape: (B, Dx_p, Dy_p, Dz_p)
    Returns:
        x: (B, Dx_p, Dy_p, Dz_p, C)
    """
    B, Dx_p, Dy_p, Dz_p = original_shape
    wx, wy, wz = window_size
    Dx_blocks, Dy_blocks, Dz_blocks = Dx_p//wx, Dy_p//wy, Dz_p//wz
    x = windows.view(
        B,
        Dx_blocks, Dy_blocks, Dz_blocks,
        wx, wy, wz, -1
    )
    x = x.permute(0,1,4,2,5,3,6,7).contiguous().view(
        B, Dx_p, Dy_p, Dz_p, -1
    )
    return x


# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


# PEG - position generating module
class PEG(nn.Module):
    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, "b ... d -> b d ...")

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)

        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")

        return x.reshape(orig_shape)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        device = torch.device("cuda")
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1")
        )
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]
        device = torch.device("cuda")
        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer("bias", bias, persistent=False)

        return self.bias


class ContinuousPositionBias(nn.Module):
    """from https://arxiv.org/abs/2111.09883"""

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims=3,  # 2 for 2D slices, 3 for 3D scan
        layers=2,
        log_dist=True,
        cache_rel_pos=False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions, device=torch.device("cpu")):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            device = torch.device("cuda")
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer("rel_pos", rel_pos, persistent=False)

        rel_pos = self.rel_pos.to(torch.float32)

        for layer in self.net:
            rel_pos = layer(rel_pos.float())

        return rearrange(rel_pos, "i j h -> h i j")


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=DIM_HEAD,
        heads=NUM_HEADS,
        causal=False,
        num_null_kv=0,
        norm_context=True,
        dropout=0.0,
        scale=8,
        window_size=(8, 8, 8),
        # window_size=None,
        pos_bias=ContinuousPositionBias(dim=256,
                                        heads=NUM_HEADS,
                                        num_dims=3),
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        self.window_size = window_size
        self.pos_bias = pos_bias

    def forward(self, x, mask=None, context=None, attn_bias=None, scan_shape=None, token_mask=None):
        # token_mask = torch.tensor([[0] + [1]*(x.shape[1] - 1)], device=x.device, dtype=torch.bool).repeat(x.shape[0], 1)
        if self.window_size is None:
            batch, device, dtype = x.shape[0], x.device, x.dtype
            device = torch.device("cuda")
            if exists(context):
                context = self.context_norm(context)

            kv_input = default(context, x)

            x = self.norm(x)

            q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
            )

            nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(
                dim=-2
            )

            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

            # bottleneck of attention, O(n^2) memory usage
            # print(f"[green]q shape: {q.shape} - [B, H, I, D][/green]")
            # print(f"[green]k shape: {k.shape} - [B, H, J, D][/green]")
            # B, H, I, _ = q.shape
            # J = k.shape[2]
            # n_elements = B * H * I * J
            # memory_MB = n_elements * 4 / 1e6       # if 32-bit floats (4 bytes) used
            # print(f"[yellow]Expecting einsum output shape: ({B}, {H}, {I}, {J})[/yellow]")
            # print(f"[red]Total elements: {n_elements:,} ({B}*{H}*{I}*{J})[/red]")
            # print(f"""[red]Estimated memory: {memory_MB:.2f} MB, total peak memory can be x5 larger than einsum output[/red]\n""")
            sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

            i, j = sim.shape[-2:]

            if exists(attn_bias):      
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
                sim = sim + attn_bias

            if exists(mask):
                mask = F.pad(mask, (self.num_null_kv, 0), value=True)
                mask = rearrange(mask, "b j -> b 1 1 j")
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
                
            # Aug 6, 2025: double mask for queries and keys
            if exists(token_mask):
                # Pad for null_kv if needed
                token_mask_padded = F.pad(token_mask, (self.num_null_kv, 0), value=True)

                # Mask keys (prevent being attended to)
                key_mask = token_mask_padded.view(batch, 1, 1, -1)
                sim = sim.masked_fill(~key_mask, -torch.finfo(sim.dtype).max)

                # Mask queries (prevent attending out)
                query_mask = token_mask.view(batch, 1, -1, 1)
                sim = sim.masked_fill(~query_mask, -torch.finfo(sim.dtype).max)

            if self.causal:
                sim = sim + self.rel_pos_bias(sim)
                device = torch.device("cuda")
                causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(
                    j - i + 1
                )
                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim=-1)
            if exists(token_mask):
                attn = attn.masked_fill(~query_mask, 0.0)
            attn = self.attn_dropout(attn)

            out = einsum("b h i j, b h j d -> b h i d", attn, v)

            out = rearrange(out, "b h n d -> b n (h d)")
            return self.to_out(out)
        
        # Else, do windowed attention
        # 1. Reshape x to (B, Dx, Dy, Dz, C)
        B = x.shape[0]
        Dx, Dy, Dz = scan_shape[1:]
        C = x.shape[-1]
        x_3d = x.view(B, Dx, Dy, Dz, C)

        # 6 Aug, 2025: double mask for queries and keys
        if token_mask is not None:
            token_mask_3d = token_mask.view(B, Dx, Dy, Dz)
            window_masks, _ = window_partition(token_mask_3d.unsqueeze(-1), self.window_size)  # (B_w, Nw, 1)
            window_masks = window_masks.squeeze(-1).to(torch.bool)  # (B_w, Nw)
        else:
            window_masks = None

        # 2. Partition into windows
        windows, (Dx_p, Dy_p, Dz_p) = window_partition(x_3d, self.window_size)  # (num_windows*B, window_volume, C)

        # 3. Optionally handle context (cross-attention)
        if context is not None:
            context_3d = context.view(B, Dx, Dy, Dz, C)  # assumes same layout
            context_windows, _ = window_partition(context_3d, self.window_size)
        else:
            context_windows = None

        # 4. For each window: do self/cross attention (as in original code, but batched)
        # We treat each window as a batch element
        # out_windows: (num_windows*B, window_volume, C)
        if self.pos_bias is not None:
            wx, wy, wz = self.window_size
            window_attn_bias = self.pos_bias(wx, wy, wz, device=x.device)  # (heads, Nw, Nw)
            out_windows = self._window_attention_forward(
                windows, context_windows, mask, window_attn_bias, token_mask=window_masks
            )
        else:
            out_windows = self._window_attention_forward(
                windows, context_windows, mask, attn_bias=None, token_mask=window_masks
            )

        # 5. Reverse windows to original layout
        out_3d = window_reverse(out_windows, self.window_size, (B, Dx_p, Dy_p, Dz_p))
        # Crop if padded
        out_3d = out_3d[:, :Dx, :Dy, :Dz, :]
        # 6. Flatten back to (B, num_blocks, C)
        out = out_3d.reshape(B, Dx*Dy*Dz, C)
        return out

    def _window_attention_forward(self, windows, context_windows, mask, attn_bias, token_mask):
        # windows: (batch_w, Nw, C)
        # context_windows: (batch_w, Nw, C) or None
        x = windows  # (batch_w, Nw, C)
        if context_windows is not None:
            kv_input = context_windows
        else:
            kv_input = x

        # Norm
        x = self.norm(x)
        kv_input = self.context_norm(kv_input)
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        B_w, Nw, _ = q.shape
        q = q.view(B_w, Nw, self.heads, -1).transpose(1,2)   # (B_w, heads, Nw, dim_head)
        k = k.view(B_w, Nw, self.heads, -1).transpose(1,2)
        v = v.view(B_w, Nw, self.heads, -1).transpose(1,2)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # Optional mask or bias
        # (can be windowed or broadcasted, up to you)
        if attn_bias is not None:
            sim = sim + attn_bias.unsqueeze(0)
        if mask is not None:
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        # Aug 6, 2025: double mask for queries and keys
        if token_mask is not None:
            key_mask = token_mask.view(B_w, 1, 1, -1)   # [B_w, 1, 1, Nw]
            query_mask = token_mask.view(B_w, 1, -1, 1) # [B_w, 1, Nw, 1]
            sim = sim.masked_fill(~key_mask, -torch.finfo(sim.dtype).max)
            sim = sim.masked_fill(~query_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        if exists(token_mask):
            attn = attn.masked_fill(~query_mask, 0.0)
        attn = self.attn_dropout(attn)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.transpose(1,2).reshape(B_w, Nw, -1)
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context=None,
        causal=False,
        dim_head=DIM_HEAD,
        heads=NUM_HEADS,
        ff_mult=4,
        peg=False,
        peg_causal=False,
        attn_num_null_kv=2,
        has_cross_attn=False,
        attn_dropout=ATTN_DROPOUT,
        ff_dropout=FF_DROPOUT
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            causal=causal,
                            dropout=attn_dropout,
                        ),
                        (
                            Attention(
                                dim=dim,
                                dim_head=dim_head,
                                dim_context=dim_context,
                                heads=heads,
                                causal=False,
                                num_null_kv=attn_num_null_kv,
                                dropout=attn_dropout,
                            )
                            if has_cross_attn
                            else None
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        scan_shape: Tuple[int, int, int, int] = None,
        attn_bias=None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None,
        token_mask=None,
    ):

        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape=scan_shape) + x
            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask, scan_shape=scan_shape, token_mask=token_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask, scan_shape=scan_shape, token_mask=token_mask) + x

            x = ff(x) + x

        return self.norm_out(x)
