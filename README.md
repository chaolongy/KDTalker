<div align="center">
<img src='https://github.com/user-attachments/assets/3fdf69a7-e2db-4c61-aad0-109e6ccc51fa' width='600px'/>
    
# Unlock Pose Diversity: Accurate and Efficient Implicit Keypoint-based Spatiotemporal Diffusion for Audio-driven Talking Portrait
[![arXiv](https://img.shields.io/badge/arXiv-KDTalker-9065CA.svg?logo=arXiv)](https://arxiv.org/abs/2503.12963)
[![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![GitHub Stars](https://img.shields.io/github/stars/chaolongy/KDTalker?style=social)](https://github.com/chaolongy/KDTalker)

<div>
    <a href='https://chaolongy.github.io/' target='_blank'>Chaolong Yang <sup>1,3*</sup> </a>&emsp;
    <a href='https://kaiseem.github.io/' target='_blank'>Kai Yao <sup>2*</a>&emsp;
    <a href='https://scholar.xjtlu.edu.cn/en/persons/YuyaoYan' target='_blank'>Yuyao Yan <sup>3</sup> </a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=HDO58yUAAAAJ' target='_blank'>Chenru Jiang <sup>4</sup> </a>&emsp;
    <a href='https://weiguangzhao.github.io/' target='_blank'>Weiguang Zhao <sup>1,3</sup> </a>&emsp; </br>
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=c-x5M2QAAAAJ' target='_blank'>Jie Sun <sup>3†</sup> </a>&emsp;
    <a href='https://sites.google.com/view/guangliangcheng' target='_blank'>Guangliang Cheng <sup>1</sup> </a>&emsp;
    <a href='https://scholar.google.com/schhp?hl=zh-CN' target='_blank'>Yifei Zhang <sup>5</sup> </a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=JNRMVNYAAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Bin Dong <sup>4</sup> </a>&emsp;
    <a href='https://sites.google.com/view/kaizhu-huang-homepage/home' target='_blank'>Kaizhu Huang <sup>4†</sup> </a>&emsp;
</div>
<br>
    
<div>
    <sup>1</sup> University of Liverpool &emsp; <sup>2</sup> Ant Group &emsp; <sup>3</sup> Xi’an Jiaotong-Liverpool University &emsp; </br>
    <sup>4</sup> Duke Kunshan University &emsp; <sup>5</sup> Ricoh Software Research Center &emsp;
</div>


<div align="justify">

# News
[2025.09.03] Our paper was accepted by the **International Journal of Computer Vision (IJCV)**.

[2025.07.30] Training and evaluation codes have been released.

[2025.07.03] Our demo **KDTalker++** was accepted by the **2025 ACM Multimedia Demo and Video Track**.

[2025.05.26] Important update! New models and new functions have been updated to local deployment [`KDTalker`](https://kdtalker.com/). New functions include background replacement and expression editing. 

[2025.04.13] A more powerful TTS has been updated in our local deployment [`KDTalker`](https://kdtalker.com/).

[2025.03.14] Release paper version demo and inference code.


# Comparative videos
https://github.com/user-attachments/assets/08ebc6e0-41c5-4bf4-8ee8-2f7d317d92cd


# Demo
Local deployment(4090) demo [`KDTalker`](https://kdtalker.com/). 

You can also visit the demo deployed on [`Huggingface`](https://huggingface.co/spaces/ChaolongYang/KDTalker), where inference is slower due to ZeroGPU.

<img width="2789" height="1553" alt="Demo" src="https://github.com/user-attachments/assets/387c9cab-4d79-48b2-96d7-f9271fe9f1d6" />


# Environment
Our KDTalker could be conducted on one RTX4090 or RTX3090.

### 1. Clone the code and prepare the environment

**Note:** Make sure your system has [`git`](https://git-scm.com/), [`conda`](https://anaconda.org/anaconda/conda), and [`FFmpeg`](https://ffmpeg.org/download.html) installed.

```
git clone https://github.com/chaolongy/KDTalker
cd KDTalker

# create env using conda
conda create -n KDTalker python=3.9
conda activate KDTalker

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

### 2. Download pretrained weights

First, you can download all LiverPorait pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib). Unzip and place them in `./pretrained_weights`.
Ensuring the directory structure is as follows:
```text
pretrained_weights
├── insightface
│   └── models
│       └── buffalo_l
│           ├── 2d106det.onnx
│           └── det_10g.onnx
└── liveportrait
    ├── base_models
    │   ├── appearance_feature_extractor.pth
    │   ├── motion_extractor.pth
    │   ├── spade_generator.pth
    │   └── warping_module.pth
    ├── landmark.onnx
    └── retargeting_models
        └── stitching_retargeting_module.pth
```
You can download the weights for the face detector, audio extractor and KDTalker from [Google Drive](https://drive.google.com/drive/folders/1OkfiFArUCsnkF_0tI2SCEAwVCBLSjzd6?hl=zh-CN). Put them in `./ckpts`.

OR, you can download above all weights in [Huggingface](https://huggingface.co/ChaolongYang/KDTalker/tree/main).


# Training

### 1. Data processing
```
python ./dataset_process/extract_motion_dataset.py -mp4_root ./path_to_your_video_root
```

### 2. Calculate data norm
```
python ./dataset_process/cal_norm.py
```

### 3. Configure wandb and train
Please configure your own "WANDB_API_KEY" on `./config/structured.py`. Then execute the code `./main.py`
```
python main.py
```


# Inference
```
python inference.py -source_image ./example/source_image/WDA_BenCardin1_000.png -driven_audio ./example/driven_audio/WDA_BenCardin1_000.wav -output ./results/output.mp4
```


# Evaluation

### 1. Diversity

First, please download the Hopenet pretrained weights from [Google Drive](https://drive.google.com/file/d/1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR/view). Put it in `./evaluation/deep-head-pose/`, and then execute the code `./evaluation/deep-head-pose/test_on_video_dlib.py`.
```
python test_on_video_dlib.py -video ./path_to_your_video_root
```

Finally, calculating the standard deviation.
```
python cal_std.py
```

### 2. Beat align
```
python cal_beat_align_score.py -video_root ./path_to_your_video_root
```

### 3. LSE-C and LSE-D
Please configure it as follows: [Wav2lip](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation).

# Contact
Our code is under the CC-BY-NC 4.0 license and intended solely for research purposes. If you have any questions or wish to use it for commercial purposes, please contact us at chaolong.yang@liverpool.ac.uk


# Citation
If you find this code helpful for your research, please cite:
```
@misc{yang2025kdtalker,
      title={Unlock Pose Diversity: Accurate and Efficient Implicit Keypoint-based Spatiotemporal Diffusion for Audio-driven Talking Portrait}, 
      author={Chaolong Yang and Kai Yao and Yuyao Yan and Chenru Jiang and Weiguang Zhao and Jie Sun and Guangliang Cheng and Yifei Zhang and Bin Dong and Kaizhu Huang},
      year={2025},
      eprint={2503.12963},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12963}, 
}
```


# Acknowledge
We acknowledge these works for their public code and selfless help: [SadTalker](https://github.com/OpenTalker/SadTalker), [LivePortrait](https://github.com/KwaiVGI/LivePortrait), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), [Face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [deep-head-pose](https://github.com/natanielruiz/deep-head-pose), [Bailando](https://github.com/lisiyao21/Bailando), etc.
</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chaolongy/KDTalker&type=Date)](https://www.star-history.com/#chaolongy/KDTalker&Date)
