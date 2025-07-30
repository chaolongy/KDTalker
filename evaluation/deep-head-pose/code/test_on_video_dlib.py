import sys, os, argparse
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
import torchvision
import torch
import glob
import dlib
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import hopenet, utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='evaluation/deep-head-pose/code/hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='evaluation/deep-head-pose/code/mmod_human_face_detector.dat', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video', default='')
    parser.add_argument('--out_dir', default='evaluation/deep-head-pose/code/output')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int, default=200)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=25.)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot


    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # model.cuda(gpu)
    model.cuda()

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 1

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    video_paths = glob.glob(os.path.join(args.video_path, '*.mp4'))

    done_paths = glob.glob(os.path.join(args.out_dir, '*.npy'))

    for video_path in video_paths:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_name_path = os.path.join(out_dir, name + '.npy')
        if out_name_path in done_paths:
            continue
        print('Num:', total)

        video = cv2.VideoCapture(video_path)

        # New cv2
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

        frame_num = 1

        pose_list = []

        while frame_num <= args.n_frames:
            print(frame_num)

            ret,frame = video.read()
            if ret == False:
                break

            cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # Dlib detect
            dets = cnn_face_detector(cv2_frame, 1)

            for idx, det in enumerate(dets):
                # Get x_min, y_min, x_max, y_max, conf
                x_min = det.rect.left()
                y_min = det.rect.top()
                x_max = det.rect.right()
                y_max = det.rect.bottom()
                conf = det.confidence

                if conf > 1.0:
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    x_min -= 2 * bbox_width / 4
                    x_max += 2 * bbox_width / 4
                    y_min -= 3 * bbox_height / 4
                    y_max += bbox_height / 4
                    x_min = int(max(x_min, 0))
                    y_min = int(max(y_min, 0))
                    x_max = int(min(frame.shape[1], x_max))
                    y_max = int(min(frame.shape[0], y_max))
                    # Crop image
                    img = cv2_frame[y_min:y_max,x_min:x_max]
                    img = Image.fromarray(img)

                    # Transform
                    img = transformations(img)
                    img_shape = img.size()
                    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                    img = Variable(img).cuda(gpu)

                    x, yaw, pitch, roll = model(img)

                    pose = np.concatenate((yaw.detach().cpu().numpy(), pitch.detach().cpu().numpy(), roll.detach().cpu().numpy()), 0)
                    pose_list.append(pose)
            frame_num += 1
        save_path = os.path.join(out_dir, video_path.split('/')[-1].split('.')[0] + '.npy')
        pose_all = np.stack(pose_list, 0)
        np.save(save_path, pose_all)
        video.release()
        total += 1

