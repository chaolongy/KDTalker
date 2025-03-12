import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import math
from einops_exts import check_shape, rearrange_many
from torch import Size, Tensor, nn

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


def map_positional_encoding(v: Tensor, freq_bands: Tensor) -> Tensor:
    """Map v to positional encoding representation phi(v)

    Arguments:
        v (Tensor): input features (B, IFeatures)
        freq_bands (Tensor): frequency bands (N_freqs, )

    Returns:
        phi(v) (Tensor): fourrier features (B, 3 + (2 * N_freqs) * 3)
    """
    pe = [v]
    for freq in freq_bands:
        fv = freq * v
        pe += [torch.sin(fv), torch.cos(fv)]
    return torch.cat(pe, dim=-1)

class FeatureMapping(nn.Module):
    """FeatureMapping nn.Module

    Maps v to features following transformation phi(v)

    Arguments:
        i_dim (int): input dimensions
        o_dim (int): output dimensions
    """

    def __init__(self, i_dim: int, o_dim: int) -> None:
        super().__init__()
        self.i_dim = i_dim
        self.o_dim = o_dim

    def forward(self, v: Tensor) -> Tensor:
        """FeratureMapping forward pass

        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): mapped features (B, OFeatures)
        """
        raise NotImplementedError("Forward pass not implemented yet!")

class PositionalEncoding(FeatureMapping):
    """PositionalEncoding module

    Maps v to positional encoding representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        N_freqs (int): #frequency to sample (default: 10)
    """

    def __init__(
        self,
        i_dim: int,
        N_freqs: int = 10,
    ) -> None:
        super().__init__(i_dim, 3 + (2 * N_freqs) * 3)
        self.N_freqs = N_freqs

        a, b = 1, self.N_freqs - 1
        freq_bands = 2 ** torch.linspace(a, b, self.N_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, v: Tensor) -> Tensor:
        """Map v to positional encoding representation phi(v)

        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourrier features (B, 3 + (2 * N_freqs) * 3)
        """
        return map_positional_encoding(v, self.freq_bands)

class BaseTemperalPointModel(nn.Module):
    """ A base class providing useful methods for point cloud processing. """

    def __init__(
        self,
        *,
        num_classes,
        embed_dim,
        extra_feature_channels,
        dim: int = 768,
        num_layers: int = 6
    ):
        super().__init__()

        self.extra_feature_channels = extra_feature_channels
        self.timestep_embed_dim = 256
        self.output_dim = num_classes
        self.dim = dim
        self.num_layers = num_layers


        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, self.timestep_embed_dim ),
            nn.SiLU(),
            nn.Linear(self.timestep_embed_dim , self.timestep_embed_dim )
        )

        self.positional_encoding = PositionalEncoding(i_dim=3, N_freqs=10)
        positional_encoding_d_out = 3 + (2 * 10) * 3

        # Input projection (point coords, point coord encodings, other features, and timestep embeddings)

        self.input_projection = nn.Linear(
            in_features=(3 + positional_encoding_d_out),
            out_features=self.dim
        )#b f p c

        # Transformer layers
        self.layers = self.get_layers()

        # Output projection
        self.output_projection = nn.Linear(self.dim, self.output_dim)
    def get_layers(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError('This method should be implemented by subclasses')
