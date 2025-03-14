# -*- coding: UTF-8 -*-
import os
os.environ['HYDRA_FULL_ERROR']='1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import shutil
import uuid
import os
import numpy as np
from tqdm import tqdm
import cv2
from rich.progress import track
import tyro


from PIL import Image
import time
import torch
import torch.nn.functional as F
from torch import nn
import imageio
from pydub import AudioSegment
from pykalman import KalmanFilter


from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.camera import get_rotation_matrix
from dataset_process import audio

from dataset_process.croper import Croper


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps
    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)
    return audio_length, num_frames

def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x

        if self.use_act:
            return self.act(out)
        else:
            return out

class AudioEncoder(nn.Module):
    def __init__(self, wav2lip_checkpoint, device):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        #### load the pre-trained audio_encoder
        wav2lip_state_dict = torch.load(wav2lip_checkpoint, map_location=torch.device(device))['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)

    def forward(self, audio_sequences):
        B = audio_sequences.size(0)

        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1
        dim = audio_embedding.shape[1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))

        return audio_embedding.squeeze(-1).squeeze(-1) #B seq_len+1 512

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def dct2device(dct: dict, device):
    for key in dct:
        dct[key] = torch.tensor(dct[key]).to(device)
    return dct

def save_video_with_watermark(video, audio, save_path):
    temp_file = str(uuid.uuid4())+'.mp4'
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video, audio, temp_file)
    os.system(cmd)
    shutil.move(temp_file, save_path)

class Inferencer(object):
    def __init__(self):
        st=time.time()
        print('#'*25+'Start initialization'+'#'*25)
        self.device = 'cuda'

        from model import get_model
        self.point_diffusion = get_model()
        ckpt = torch.load('KDTalker.pth')

        self.point_diffusion.load_state_dict(ckpt['model'])
        self.point_diffusion.eval()
        self.point_diffusion.to(self.device)

        lm_croper_checkpoint = 'ckpts/shape_predictor_68_face_landmarks.dat'
        self.croper = Croper(lm_croper_checkpoint)

        self.norm_info = dict(np.load('dataset_process/norm.npz'))

        wav2lip_checkpoint = 'ckpts/wav2lip.pth'
        self.wav2lip_model = AudioEncoder(wav2lip_checkpoint, 'cuda')
        self.wav2lip_model.cuda()
        self.wav2lip_model.eval()

        # set tyro theme
        tyro.extras.set_accent_color("bright_cyan")
        args = tyro.cli(ArgumentConfig)

        # specify configs for inference
        self.inf_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
        self.crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

        self.live_portrait_pipeline = LivePortraitPipeline(inference_cfg=self.inf_cfg, crop_cfg=self.crop_cfg)

    def _norm(self, data_dict):
        for k in data_dict.keys():
            if k in ['yaw', 'pitch', 'roll', 't', 'exp', 'scale', 'kp', ]:
                v=data_dict[k]
                data_dict[k] = (v - self.norm_info[k+'_mean'])/self.norm_info[k+'_std']
        return data_dict

    def _denorm(self, data_dict):
        for k in data_dict.keys():
            if k in ['yaw', 'pitch', 'roll', 't', 'exp', 'scale', 'kp']:
                v=data_dict[k]
                data_dict[k] = v * self.norm_info[k+'_std'] + self.norm_info[k+'_mean']
        return data_dict

    def output_to_dict(self, data):
        output = {}
        output['scale'] = data[:, 0]
        output['yaw'] = data[:, 1, None]
        output['pitch'] = data[:, 2, None]
        output['roll'] = data[:, 3, None]
        output['t'] = data[:, 4:7]
        output['exp'] = data[:, 7:]
        return output

    def extract_mel_from_audio(self, audio_file_path):
        syncnet_mel_step_size = 16
        fps = 25
        wav = audio.load_wav(audio_file_path, 16000)
        wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio.melspectrogram(wav).T
        spec = orig_mel.copy()
        indiv_mels = []

        for i in tqdm(range(num_frames), 'mel:'):
            start_frame_num = i - 2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)  # T 80 16
        return indiv_mels

    def extract_wav2lip_from_audio(self, audio_file_path):
        asd_mel = self.extract_mel_from_audio(audio_file_path)
        asd_mel = torch.FloatTensor(asd_mel).cuda().unsqueeze(0).unsqueeze(2)
        with torch.no_grad():
            hidden = self.wav2lip_model(asd_mel)
        return hidden[0].cpu().detach().numpy()

    def headpose_pred_to_degree(self, pred):
        device = pred.device
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred)
        degree = torch.sum(pred * idx_tensor, 1) * 3 - 99
        return degree

    @torch.no_grad()
    def generate_with_audio_img(self, image_path, audio_path, save_path):
        image = np.array(Image.open(image_path).convert('RGB'))
        cropped_image, crop, quad = self.croper.crop([image], still=False, xsize=512)
        input_image = cv2.resize(cropped_image[0], (256, 256))

        I_s = torch.FloatTensor(input_image.transpose((2, 0, 1))).unsqueeze(0).cuda() / 255

        x_s_info = self.live_portrait_pipeline.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp'].reshape(1, 21, -1)
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)

        ######## process driving info ########
        kp_info = {}
        for k in x_s_info.keys():
            kp_info[k] = x_s_info[k].cpu().numpy()

        kp_info = self._norm(kp_info)

        ori_kp = torch.cat([torch.zeros([1, 7]), torch.Tensor(kp_info['kp'])], -1).cuda()

        input_x = np.concatenate([kp_info[k] for k in ['scale', 'yaw', 'pitch', 'roll', 't', 'exp']], 1)
        input_x = np.expand_dims(input_x, -1)
        input_x = np.expand_dims(input_x, 0)
        input_x = np.concatenate([input_x, input_x, input_x], -1)

        aud_feat = self.extract_wav2lip_from_audio(audio_path)

        sample_frame = 64
        padding_size = (sample_frame - aud_feat.shape[0] % sample_frame) % sample_frame

        if padding_size > 0:
            aud_feat = np.concatenate((aud_feat, aud_feat[:padding_size, :]), axis=0)
        else:
            aud_feat = aud_feat

        outputs = [input_x]

        sample_frame = 64
        for i in range(0, aud_feat.shape[0] - 1, sample_frame):
            input_mel = torch.Tensor(aud_feat[i: i + sample_frame]).unsqueeze(0).cuda()
            kp0 = torch.Tensor(outputs[-1])[:, -1].cuda()
            pred_kp = self.point_diffusion.forward_sample(70, ref_kps=kp0, ori_kps=ori_kp, aud_feat=input_mel,
                                                          scheduler='ddim', num_inference_steps=50)
            outputs.append(pred_kp.cpu().numpy())

        outputs = np.mean(np.concatenate(outputs, 1)[0, 1:aud_feat.shape[0] - padding_size + 1], -1)
        output_dict = self.output_to_dict(outputs)
        output_dict = self._denorm(output_dict)

        num_frame = output_dict['yaw'].shape[0]
        x_d_info = {}
        for key in output_dict:
            x_d_info[key] = torch.tensor(output_dict[key]).cuda()

        # smooth
        def smooth(sequence, n_dim_state=1):
            kf = KalmanFilter(initial_state_mean=sequence[0],
                              transition_covariance=0.05 * np.eye(n_dim_state),
                              observation_covariance=0.001 * np.eye(n_dim_state))
            state_means, _ = kf.smooth(sequence)
            return state_means

        yaw_data = x_d_info['yaw'].cpu().numpy()
        pitch_data = x_d_info['pitch'].cpu().numpy()
        roll_data = x_d_info['roll'].cpu().numpy()
        t_data = x_d_info['t'].cpu().numpy()
        exp_data = x_d_info['exp'].cpu().numpy()

        smoothed_pitch = smooth(pitch_data, n_dim_state=1)
        smoothed_yaw = smooth(yaw_data, n_dim_state=1)
        smoothed_roll = smooth(roll_data, n_dim_state=1)
        smoothed_t = smooth(t_data, n_dim_state=3)
        smoothed_exp = smooth(exp_data, n_dim_state=63)

        x_d_info['pitch'] = torch.Tensor(smoothed_pitch).cuda()
        x_d_info['yaw'] = torch.Tensor(smoothed_yaw).cuda()
        x_d_info['roll'] = torch.Tensor(smoothed_roll).cuda()
        x_d_info['t'] = torch.Tensor(smoothed_t).cuda()
        x_d_info['exp'] = torch.Tensor(smoothed_exp).cuda()

        template_dct = {'motion': [], 'c_d_eyes_lst': [], 'c_d_lip_lst': []}
        for i in track(range(num_frame), description='Making motion templates...', total=num_frame):
            x_d_i_info = x_d_info
            R_d_i = get_rotation_matrix(x_d_i_info['pitch'][i], x_d_i_info['yaw'][i], x_d_i_info['roll'][i])

            item_dct = {
                'scale': x_d_i_info['scale'][i].cpu().numpy().astype(np.float32),
                'R_d': R_d_i.cpu().numpy().astype(np.float32),
                'exp': x_d_i_info['exp'][i].reshape(1, 21, -1).cpu().numpy().astype(np.float32),
                't': x_d_i_info['t'][i].cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

        I_p_lst = []
        R_d_0, x_d_0_info = None, None

        for i in track(range(num_frame), description='🚀Animating...', total=num_frame):
            x_d_i_info = template_dct['motion'][i]
            for key in x_d_i_info:
                x_d_i_info[key] = torch.tensor(x_d_i_info[key]).cuda()
            R_d_i = x_d_i_info['R_d']

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            if self.inf_cfg.flag_relative_motion:
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                delta_new = x_s_info['exp'].reshape(1, 21, -1) + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                R_new = R_d_i
                delta_new = x_d_i_info['exp']
                scale_new = x_s_info['scale']
                t_new = x_d_i_info['t']

            t_new[..., 2].fill_(0)
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            out = self.live_portrait_pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

        video_name = save_path.split('/')[-1]
        video_save_dir = os.path.dirname(save_path)
        path = os.path.join(video_save_dir, 'temp_' + video_name)

        imageio.mimsave(path, I_p_lst, fps=float(25))

        audio_name = audio_path.split('/')[-1]
        new_audio_path = os.path.join(video_save_dir, audio_name)
        start_time = 0
        sound = AudioSegment.from_file(audio_path)
        end_time = start_time + num_frame * 1 / 25 * 1000
        word1 = sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        save_video_with_watermark(path, new_audio_path, save_path, watermark=False)
        print(f'The generated video is named {video_save_dir}/{video_name}')

        os.remove(path)
        os.remove(new_audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-source_image", type=str, default="example/source_image/WDA_BenCardin1_000.png",
                        help="source image")
    parser.add_argument("-driven_audio", type=str, default="example/driven_audio/WDA_BenCardin1_000.wav",
                        help="driving audio")
    parser.add_argument("-output", type=str, default="results/output.mp4", help="output video file name", )

    args = parser.parse_args()

    Infer = Inferencer()
    Infer.generate_with_audio_img(args.source_image, args.driven_audio, args.output)