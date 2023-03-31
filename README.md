# Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video (CVPR 2023)

This repository contains the official implementation of the CVPR 2023 paper "Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video".

### [arXiv](https://arxiv.org/abs/2303.16053) | [Video](https://youtu.be/OOhw0fb2aE4)   

## Introduction
![fig](fig1.png)

Real-time eyeblink detection in the wild can widely serve for fatigue detection, face anti-spoofing, emotion analysis, etc. The existing research efforts generally focus on single-person cases towards trimmed video. However, multi-person scenario within untrimmed videos is also important for practical applications, which has not been well concerned yet. To address this, we shed light on this research field for the first time with essential contributions on dataset, theory, and practices. In particular, a large-scale dataset termed MPEblink that involves 686 untrimmed videos with 8748 eyeblink events is proposed under multi-person conditions. The samples are captured from unconstrained films to reveal "in the wild" characteristics. Meanwhile, a real-time multi-person eyeblink detection method is also proposed. Being different from the existing counterparts, our proposition runs in a one-stage spatio-temporal way with end-to-end learning capacity. Specifically, it simultaneously addresses the sub-tasks of face detection, face tracking, and human instance-level eyeblink detection. This paradigm holds 2 main advantages: (1) eyeblink features can be facilitated via the face's global context (e.g., head pose and illumination condition) with joint optimization and interaction, and (2) addressing these sub-tasks in parallel instead of sequential manner can save time remarkably to meet the real-time running requirement. Experiments on MPEblink verify the essential challenges of real-time multi-person eyeblink detection in the wild for untrimmed video. Our method also outperforms existing approaches by large margins and with a high inference speed.

## MPEblink Dataset

![mpeblink](mpeblink.png)

Existing eyeblink detection datasets generally focus on single-person scenarios, and are also limited in aspects of constrained conditions or trimmed short videos. To explore unconstrained eyeblink detection under multi-person and untrimmed scenarios, we construct a large-scale multi-person eyeblink detection dataset termed MPEblink to shed the light on this research field that has not been well studied before. The distinguishing characteristics of MPEblink lie in 3 aspects: multi-person, unconstrained, and untrimmed long video, which makes our benchmark more realistic and challenging.

The dataset is available at [here](https://doi.org/10.5281/zenodo.7754768).

## InstBlink

InstBlink is a one-stage multi-person eyeblink detection framework that can jointly perform face detection, face tracking, and instance-level eyeblink detection. The code will be available soon.

## Citation

If you find our work useful in your research, please consider to cite our paper:

```
@inproceedings{zeng2023_mpeblink,
  title={Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video},
  author={Zeng, Wenzheng and Xiao, Yang and Wei, Sicheng and Gan, Jinfang and Zhang, Xintao and Cao, Zhiguo and Fang, Zhiwen and Zhou, Jeoy, Tianyi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

