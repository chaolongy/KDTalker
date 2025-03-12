import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from .transformer_utils import BaseTemperalPointModel
import math
from einops_exts import check_shape, rearrange_many
from functools import partial
from rotary_embedding_torch import RotaryEmbedding

def exists(x):
    return x is not None

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma + self.beta


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, heads=4, attn_head_dim=None, casual_attn=False,rotary_emb = None):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.casual_attn = casual_attn

        if attn_head_dim is not None:
            head_dim = attn_head_dim

        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.proj = nn.Linear(all_head_dim, dim)
        self.rotary_emb = rotary_emb

    def forward(self, x, pos_bias = None):
        N, device = x.shape[-2], x.device
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.num_heads)

        q = q * self.scale

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = torch.einsum('... h i d, ... h j d -> ... h i j', q, k)

        if exists(pos_bias):
            sim = sim + pos_bias

        if self.casual_attn:
            mask = torch.tril(torch.ones(sim.size(-1), sim.size(-2))).to(device)
            sim = sim.masked_fill(mask[..., :, :] == 0, float('-inf'))

        attn = sim.softmax(dim = -1)
        x = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)
        x = rearrange(x, '... h n d -> ... n (h d)')
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        self.norm = LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            x = self.norm(x)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, cond_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_out * 2)
        ) if exists(cond_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)

    def forward(self, x, cond_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(cond_emb), 'time emb must be passed in'
            cond_emb = self.mlp(cond_emb)
            #cond_emb = rearrange(cond_emb, 'b f c -> b f 1 c')
            scale_shift = cond_emb.chunk(2, dim=-1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + x

class SimpleTransModel(BaseTemperalPointModel):
    """
    A simple model that processes a point cloud by applying a series of MLPs to each point
    individually, along with some pooled global features.
    """

    def get_layers(self):
        self.input_projection = nn.Linear(
            in_features=70,
            out_features=self.dim
        )

        cond_dim = 512 + self.timestep_embed_dim

        num_head = self.dim//64

        rotary_emb = RotaryEmbedding(min(32, num_head))

        self.time_rel_pos_bias = RelativePositionBias(heads=num_head, max_distance=128)  # realistically will not be able to generate that many frames of video... yet

        temporal_casual_attn = lambda dim: Attention(dim, heads=num_head, casual_attn=False,rotary_emb=rotary_emb)

        cond_block = partial(ResnetBlock, cond_dim=cond_dim)

        layers = nn.ModuleList([])

        for _ in range(self.num_layers):
            layers.append(nn.ModuleList([
                cond_block(self.dim, self.dim),
                cond_block(self.dim, self.dim),
                Residual(PreNorm(self.dim, temporal_casual_attn(self.dim)))
            ]))

        return layers

    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor, context=None):
        """
         Apply the model to an input batch.
         :param x: an [N x C x ...] Tensor of inputs.
         :param timesteps: a 1-D batch of timesteps.
         :param context: conditioning plugged in via crossattn
         """
        # Prepare inputs

        batch, num_frames, channels = inputs.size()

        device = inputs.device
        x = self.input_projection(inputs)

        t_emb = self.time_mlp(timesteps) if exists(self.time_mlp) else None
        t_emb = t_emb[:,None,:].expand(-1, num_frames, -1)  # b f c
        if context is not None:
            t_emb = torch.cat([t_emb, context],-1)

        time_rel_pos_bias = self.time_rel_pos_bias(num_frames, device=device)

        for block1, block2,  temporal_attn in self.layers:
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)

        # Project
        x = self.output_projection(x)
        return x