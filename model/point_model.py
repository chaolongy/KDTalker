import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from torch import Tensor

from .temporaltrans.temptrans import SimpleTransModel

class PointModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        model_type: str = 'pvcnn',
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        dropout: float = 0.1,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
    ):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'simple':
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = SimpleTransModel(
                embed_dim=embed_dim,
                num_classes=out_channels,
                extra_feature_channels=(in_channels - 3),
            )
            self.model.output_projection.bias.data.normal_(0, 1e-6)
            self.model.output_projection.weight.data.normal_(0, 1e-6)
        else:
            raise NotImplementedError()

    def forward(self, inputs: Tensor, t: Tensor, context=None) -> Tensor:
        """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """
        with self.autocast_context:
            return self.model(inputs, t, context)
