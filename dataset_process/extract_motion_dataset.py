import glob
import numpy as np
import audio
import tyro
import cv2
import concurrent.futures
import argparse

from wav2lip import *
from scipy.io import loadmat, savemat
from tqdm import tqdm

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def extract_frames_from_mp4(mp4_file_path):
    video = cv2.VideoCapture(mp4_file_path)
    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    video.release()
    return frames

def extract_motion_from_frames(frames, model):

    output_dict = {'yaw': [], 'pitch': [], 'roll': [], 't': [], 'exp': [], 'scale': [], 'kp': []}

    for frame in frames:
        input = torch.FloatTensor(frame.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).cuda() / 255
        kp_info = model.live_portrait_wrapper.get_kp_info(input)

        for k in kp_info.keys():
            output_dict[k].append(kp_info[k].cpu().numpy())
    for k in output_dict.keys():
        output_dict[k] = np.concatenate(output_dict[k], 0)
    return output_dict

def extract_wav2lip_from_mp4(mp4_file_path):
    asd_mel = extract_mel_from_mp4(mp4_file_path)
    asd_mel = torch.FloatTensor(asd_mel).cuda().unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        hidden = wav2lip_model(asd_mel)
    return hidden[0].cpu().detach().numpy()

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

def extract_mel_from_mp4(mp4_file_path):
    syncnet_mel_step_size = 16
    fps = 25
    audio_file_path = mp4_file_path.split('/')[-1].replace('.mp4', '.wav')
    command = f'ffmpeg -i {mp4_file_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_file_path}'
    os.system(command)
    wav = audio.load_wav(audio_file_path, 16000)
    os.remove(audio_file_path)
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()
    indiv_mels = []

    for i in tqdm(range(num_frames), 'mel:'):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)         # T 80 16
    return indiv_mels

def process_video(fp, model, dst):
    print('process phase 1:', fp)
    frames = extract_frames_from_mp4(fp)

    print('process phase 2:', fp)
    motion = extract_motion_from_frames(frames, model)

    print('process phase 3:', fp)
    aud_feat = extract_wav2lip_from_mp4(fp)

    motion['aud_feat'] = aud_feat

    savemat(dst, motion)

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    max_threads = 8
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)
    live_portrait_pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp4_root", type=str, required=True, help="videos root")
    parser.add_argument("-output_root", type=str, default="datasets/motion_wav2lip", help="output motion root", )
    args = parser.parse_args()

    video_paths = sorted(glob.glob(os.path.join(args.mp4_root, '*.mp4')))

    os.makedirs(args.output_root, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        for video_path in video_paths:
            id = video_path.split('/')[-1]
            output_path = os.path.join(args.output_root, id.replace('mp4', 'mat'))
            if os.path.isfile(output_path):
                continue
            executor.submit(process_video, video_path, live_portrait_pipeline, output_path)
