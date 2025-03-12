import inspect
from typing import Optional
from einops import rearrange
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler

from torch import Tensor
from tqdm import tqdm
from diffusers import ModelMixin
from .model_utils import get_custom_betas
from .point_model import PointModel
import copy
import torch.nn as nn

class TemporalSmoothnessLoss(nn.Module):
    def __init__(self):
        super(TemporalSmoothnessLoss, self).__init__()

    def forward(self, input):
        # Calculate the difference between consecutive frames
        diff = input[:, 1:, :] - input[:, :-1, :]

        # Compute the L2 norm (squared) of the differences
        smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=2))

        return smoothness_loss

class ConditionalPointCloudDiffusionModel(ModelMixin):
    def __init__(
        self,
        beta_start: float = 1e-5,
        beta_end: float = 8e-3,
        beta_schedule: str = 'linear',
        point_cloud_model: str = 'simple',
        point_cloud_model_embed_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = 70  # 3 for 3D point positions
        self.out_channels = 70

        # Checks
        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddim']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_model = PointModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def forward_train(
        self,
        pc: Optional[Tensor],
        ref_kps: Optional[Tensor],
        ori_kps: Optional[Tensor],
        aud_feat: Optional[Tensor],
        mode: str = 'train',
        return_intermediate_steps: bool = False
    ):

        # Normalize colors and convert to tensor
        x_0 = pc
        B, Nf, Np, D = x_0.shape# batch, nums of frames, nums of points, 3


        x_0=x_0[:,:,:,0]# batch, nums of frames, 70

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,),
            device=self.device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # Conditioning
        ref_kps = ref_kps[:, :, 0]

        x_t_input = torch.cat([ori_kps.unsqueeze(1), ref_kps.unsqueeze(1), x_t], dim=1)

        aud_feat = torch.cat([torch.zeros(B, 2, 512).cuda(), aud_feat], 1)

        # Augmentation for audio feature
        if mode in 'train':
            if torch.rand(1) > 0.3:
                mean = torch.mean(aud_feat)
                std = torch.std(aud_feat)
                sample = torch.normal(mean=torch.full(aud_feat.shape, mean), std=torch.full(aud_feat.shape, std)).cuda()
                aud_feat = sample + aud_feat
            else:
                pass
        else:
            pass

        # Forward
        noise_pred = self.point_model(x_t_input, timestep, context=aud_feat)    #torch.cat([mel_feat,style_embed],-1))
        noise_pred = noise_pred[:, 2:]

        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

        loss = F.mse_loss(noise_pred, noise)

        loss_pose = F.mse_loss(noise_pred[:, :, 1:7], noise[:, :, 1:7])
        loss_exp = F.mse_loss(noise_pred[:, :, 7:], noise[:, :, 7:])


        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        return loss, loss_exp, loss_pose

    @torch.no_grad()
    def forward_sample(
        self,
        num_points: int,
        ref_kps: Optional[Tensor],
        ori_kps: Optional[Tensor],
        aud_feat: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 50,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):

        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        Np = num_points
        Nf = aud_feat.size(1)
        B = 1
        D = 3
        device = self.device

        # Sample noise
        x_t = torch.randn(B, Nf, Np, D, device=device)

        x_t = x_t[:, :, :, 0]

        ref_kps = ref_kps[:,:,0]

        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        aud_feat = torch.cat([torch.zeros(B, 2, 512).cuda(), aud_feat], 1)

        for i, t in enumerate(progress_bar):
            x_t_input = torch.cat([ori_kps.unsqueeze(1).detach(),ref_kps.unsqueeze(1).detach(), x_t], dim=1)

            # Forward
            noise_pred = self.point_model(x_t_input, t.reshape(1).expand(B), context=aud_feat)[:, 2:]

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = x_t
        output = torch.stack([output,output,output],-1)
        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
        return (output, all_outputs) if return_all_outputs else output

    def forward(self, batch: dict, mode: str = 'train', **kwargs):
        """A wrapper around the forward method for training and inference"""

        if mode == 'train':
            return self.forward_train(
                pc=batch['sequence_keypoints'],
                ref_kps=batch['ref_keypoint'],
                ori_kps=batch['ori_keypoint'],
                aud_feat=batch['aud_feat'],
                mode='train',
                **kwargs)
        elif mode == 'val':
            return self.forward_train(
                pc=batch['sequence_keypoints'],
                ref_kps=batch['ref_keypoint'],
                ori_kps=batch['ori_keypoint'],
                aud_feat=batch['aud_feat'],
                mode='val',
                **kwargs)
        elif mode == 'sample':
            num_points = 70
            return self.forward_sample(
                num_points=num_points,
                ref_kps=batch['ref_keypoint'],
                ori_kps=batch['ori_keypoint'],
                aud_feat=batch['aud_feat'],
                **kwargs) 
        else:
            raise NotImplementedError()