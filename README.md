# Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video (CVPR 2023)
[Wenzheng Zeng](https://wenzhengzeng.github.io/)<sup>1</sup>, [Yang Xiao](https://scholar.google.com/citations?user=NeKBuXEAAAAJ)<sup>1†</sup>, Sicheng Wei<sup>1</sup>, Jinfang Gan<sup>1</sup>, Xintao Zhang<sup>1</sup>, [Zhiguo Cao](https://scholar.google.com/citations?user=396o2BAAAAAJ)<sup>1</sup>, [Zhiwen Fang](https://scholar.google.com/citations?user=UX5N_FQAAAAJ)<sup>2</sup>, [Joey Tianyi Zhou](https://joeyzhouty.github.io/)<sup>3</sup>.

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Southern Medical University, <sup>3</sup>A*STAR.

<img src="pictures/demo1.gif" width="50%"/><img src="pictures/demo2.gif" width="50%"/>

### [arXiv](https://arxiv.org/abs/2303.16053) | [Video](https://www.youtube.com/watch?v=ngME7dym0Uk) 

This repository contains the official implementation of the CVPR 2023 paper "Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video".


## Introduction
<div align="center">
<img src="pictures/fig1.png" width="95%"/>
</div>

Real-time eyeblink detection in the wild can widely serve for fatigue detection, face anti-spoofing, emotion analysis, etc. The existing research efforts generally focus on single-person cases towards trimmed video. However, multi-person scenario within untrimmed videos is also important for practical applications, which has not been well concerned yet. To address this, we shed light on this research field for the first time with essential contributions on dataset, theory, and practices. In particular, a large-scale dataset termed MPEblink that involves 686 untrimmed videos with 8748 eyeblink events is proposed under multi-person conditions. The samples are captured from unconstrained films to reveal "in the wild" characteristics. Meanwhile, a real-time multi-person eyeblink detection method is also proposed. Being different from the existing counterparts, our proposition runs in a one-stage spatio-temporal way with end-to-end learning capacity. Specifically, it simultaneously addresses the sub-tasks of face detection, face tracking, and human instance-level eyeblink detection. This paradigm holds 2 main advantages: (1) eyeblink features can be facilitated via the face's global context (e.g., head pose and illumination condition) with joint optimization and interaction, and (2) addressing these sub-tasks in parallel instead of sequential manner can save time remarkably to meet the real-time running requirement. Experiments on MPEblink verify the essential challenges of real-time multi-person eyeblink detection in the wild for untrimmed video. Our method also outperforms existing approaches by large margins and with a high inference speed.

## MPEblink Dataset

<img src="pictures/mpeblink.png" width="95%"/>

Existing eyeblink detection datasets generally focus on single-person scenarios, and are also limited in aspects of constrained conditions or trimmed short videos. To explore unconstrained eyeblink detection under multi-person and untrimmed scenarios, we construct a large-scale multi-person eyeblink detection dataset termed MPEblink to shed the light on this research field that has not been well studied before. The distinguishing characteristics of MPEblink lie in 3 aspects: multi-person, unconstrained, and untrimmed long video, which makes our benchmark more realistic and challenging.

The dataset is available at [here](https://doi.org/10.5281/zenodo.7754768).

## InstBlink

InstBlink is a one-stage multi-person eyeblink detection framework that can jointly perform face detection, face tracking, and instance-level eyeblink detection.

### Installation

1. Create a new conda environment:

   ```bash
   conda create -n instblink python=3.9
   conda activate instblink
   ```
   
2. Install Pytorch (1.7.1 is recommended), scipy, tqdm, pandas.

3. Install MMDetection. 

   * Install [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) first. 1.4.8 is recommended.

   * ```bash
     cd MPEblink
     pip install -v -e .
     ```

### Data preparation

1. Download the [MPEblink dataset](https://doi.org/10.5281/zenodo.7754768). Remember to change the dataset root path into yours in `configs/base/mpeblink.py`.

2. Convert the videos to raw frames.
    ```bash
    python tools/dataset_converters/mpeblink_build_raw_frames_dataset.py --root $YOUR_DATA_PATH
    ```

### Demo

You can put some videos in `demo_video/source_video/` and get the visualization inference result in `demo_video/visual_result/` by running the following command:

  ```bash
  bash tools/code_for_demo/demo.sh
  ```


### Inference & evaluation

* You can download the pre-trained model at [Google Drive](https://drive.google.com/file/d/1kRx_pPpOwAk9D6O3M5Ed7vyAqkZbCh83/view?usp=sharing) or [Baidu Drive (code avk9)](https://pan.baidu.com/s/1UxZ7PDc76wc5y3n5QUqFqg) and put it in the `pretrained_models` directory.

* Run `test.sh` for inference and evaluation. Remember to change the dataset path into yours.

  ```bash
  bash tools/test_eval.sh
  ```


### Training

* Download the pretrained [tevit_r50](https://github.com/hustvl/Storage/releases/download/v1.1.0/tevit_r50.pth) model and place it in the `pretrained_models` directory.

* Run `train.sh` to begin training.

  ```bash
  bash tools/train.sh
  ```

## Acknowledgement

This code is inspired by [TeViT](https://github.com/hustvl/TeViT) and [MMDetection](https://github.com/open-mmlab/mmdetection). Thanks for their great contributions on the computer vision community.

## Citation

If you find our work useful in your research, please consider to cite our paper:

  ```
@inproceedings{zeng2023_mpeblink,
  title={Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video},
  author={Zeng, Wenzheng and Xiao, Yang and Wei, Sicheng and Gan, Jinfang and Zhang, Xintao and Cao, Zhiguo and Fang, Zhiwen and Zhou, Joey, Tianyi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
  ```

