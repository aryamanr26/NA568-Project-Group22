[comment]: <> (# SuPerMVO: SuperPoint-Glued Pose Estimation for Monocular Visual Odometry)

<p align="center">
  <h2 align="center">SuPerMVO: SuperPoint-Glued Pose Estimation for 
    Monocular Visual Odometry</h2>
  <p align="center">
    <a href="https://www.linkedin.com/in/aaron-sequeira"><strong>Aaron Sequeria*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/aryamanrao26"><strong>Aryaman Rao*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/kunal-siddhawar-858839140"><strong>Kunal Siddhawar*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/kush-patel-5397281b8"><strong>Kush Patel*</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>

 <!--## SuPerMVO: SuperPoint-Glued Pose Estimation for Monocular Visual Odometry -->

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![KITTI](https://img.shields.io/badge/Dataset-KITTI-green)

---

### 👨‍💻 Authors
- [Aaron Sequeira](https://www.linkedin.com/in/aaronsequeira/)
- [Aryaman Rao](https://www.linkedin.com/in/aryaman-rao/)
- [Kush Patel](https://www.linkedin.com/in/kushpatel19/)
- [Kunal Siddhawar](https://www.linkedin.com/in/kunalsiddhawar/)

---

📄 [**Paper Link**](https://link-to-your-paper.com)  
🖼️ **YouTube**  
🖼️ **Poster Preview**  
<img src="media/poster_graph_final.png" alt="SuPerMVO Poster" width="700"/>


---

### 📌 Project Summary

**SuPerMVO** is a monocular visual odometry pipeline that integrates deep learning-based feature detection and matching using SuperPoint and SuperGlue, followed by SE(3)-based pose estimation and optimization via GTSAM. Evo toolkit is used for scale correction and evaluation against KITTI ground truth. The system performs robustly even in texture-sparse and dynamic scenes, outperforming traditional ORB-SLAM-based methods in both translational and rotational accuracy.

---

## 🚀 Getting Started

### ⚙️ Create Conda Environment

```bash
conda create -n supermvo python=3.9
conda activate supermvo
```

### 🔍 Check CUDA version
```bash
nvcc --version
```

### 🔧 Install PyTorch (match to your CUDA version)
```bash
# For CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 📚 KITTI Dataset
We use the KITTI Odometry dataset for evaluation. You can download the ground truth and image sequences from the official site: [KITTI Odometry Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

- Download the odometry dataset (grayscal, 22GB)
- Download odometry ground truth poses (4 MB)
- Inside main.py, add the path for ground truth poses and images in variable groundtruth_file and image_folder respectively.

## 📚 Pre-trained Model
clone this repo in working directory: [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) 

