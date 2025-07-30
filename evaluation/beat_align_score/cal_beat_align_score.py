import glob

import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter as G
import librosa
from moviepy.editor import VideoFileClip
import mediapipe as mp
import os
import argparse


def calc_motion_beats(keypoints):
    keypoints = np.array(keypoints)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)[0]  # 提取索引
    return motion_beats, len(kinetic_vel)

def beat_align_score(B_m, B_d, sigma):
    B_m = np.array(B_m)
    B_d = np.array(B_d)
    score = 0
    for t_m in B_m:
        distances = np.abs(B_d - t_m)
        min_distance = np.min(distances)
        score += np.exp(- (min_distance ** 2) / (2 * sigma ** 2))
    score /= len(B_m)
    return score

def extract_audio_beats(video_path):
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path)
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    os.remove(audio_path)
    return beat_times, sr

landmark_68_indices = [
     10,  109,  67, 103,  54,  21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 389, 253, 450, 447,
    376, 433, 432, 287, 285, 295, 282, 276, 283, 278, 295, 197, 332, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 91, 146, 61, 185, 40, 39, 37, 0, 267, 269
]

def extract_head_motion(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    keypoints = []

    clip = VideoFileClip(video_path)
    fps = clip.fps
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        for frame in clip.iter_frames():
            results = face_mesh.process(frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    filtered_landmarks = [(face_landmarks.landmark[i].x,
                                           face_landmarks.landmark[i].y,
                                           face_landmarks.landmark[i].z) for i in landmark_68_indices]
                    keypoints.append(filtered_landmarks)

    return keypoints, fps

def main(video_path, sigma=0.1):
    beat_times, sr = extract_audio_beats(video_path)

    keypoints, fps = extract_head_motion(video_path)
    motion_beats, num_frames = calc_motion_beats(keypoints)

    motion_beats_times = motion_beats / fps

    score = beat_align_score(beat_times, motion_beats_times, sigma)

    print(f"Beat Align Score: {score}")
    return score

# 运行主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-video_root", type=str, default="")
    parser.add_argument("-output_txt_name", type=str, default='result_GT.txt')

    args = parser.parse_args()

    video_paths = sorted(glob.glob(os.path.join(args.video_root, '*.mp4')))
    scores = []
    for video_path in video_paths:
        score = main(video_path)
        scores.append(score)
        with open(args.output_txt_name, "a+") as result_file:
            result_file.write(f"{video_path.split('/')[-1]}: {score}\n")

    BA = np.mean(scores)
    with open(args.output_txt_name, "a+") as result_file:
        result_file.write(f"Mean Beat Align Score: {BA}\n")
    print(f"Mean Beat Align Score: {BA}")


