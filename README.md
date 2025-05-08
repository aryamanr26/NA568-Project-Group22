<p align="center">
  <h1 align="center">SuPerMVO: SuperPoint-Glued Pose Estimation for Monocular Visual Odometry</h1>

  <p align="center">
    <a href="https://www.linkedin.com/in/aaron-sequeira/"><strong>Aaron Sequeira</strong></a> ·
    <a href="https://www.linkedin.com/in/aryamanrao26/"><strong>Aryaman Rao</strong></a> ·
    <a href="https://www.linkedin.com/in/kush-patel-5397281b8/"><strong>Kush Patel</strong></a> ·
    <a href="https://www.linkedin.com/in/kunal-siddhawar-858839140/"><strong>Kunal Siddhawar</strong></a>
  </p>
  
  <h3 align="center">
    <a href="media/SuPerMVO.pdf">Paper</a> |
    <a href="https://www.youtube.com/watch?v=dF_nQ6IA1po">YouTube</a>
    <!-- <a href="media/ROB530_Poster.JPG">Poster</a> -->
  </h3>

  <p align="center">
    <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6-red" />
    <img alt="KITTI" src="https://img.shields.io/badge/Dataset-KITTI-green" />
  </p>

  <p align="center">
    <img src="./media/ROB530_Poster.jpg" alt="SuPerMVO Poster" width="850"/>
  </p>
</p>

---

### 📌 Project Summary

**SuPerMVO** is a monocular visual odometry pipeline that integrates deep learning-based feature detection and matching using [SuperPoint](https://github.com/rpautrat/SuperPoint) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), followed by SE(3)-based pose estimation and graph optimization via GTSAM. For scale correction and quantitative evaluation, we employ the [evo](https://github.com/MichaelGrupp/evo) toolkit in conjunction with the KITTI odometry benchmark. Our method performs robustly in low-texture and dynamic environments, significantly outperforming traditional ORB-SLAM pipelines in both translational and rotational accuracy.

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
conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 🔧 Install Other Dependancy
```bash
# Install conda packages
conda install pillow=11.0.0 pandas=2.2.3 opencv=4.11.0 scipy=1.13.1 numpy=1.26.4 -c conda-forge

# Install GTSAM
conda install -c conda-forge gtsam=4.2.0

# Install HuggingFace Transformers
pip install transformers==4.50.3
```

### 📚 KITTI Dataset
We evaluate our pipeline on the KITTI Odometry Benchmark, leveraging both the raw image sequences and the official ground‑truth trajectories. To set up:

1. Download the **Grayscale Odometry Sequences** (≈22 GB) and the corresponding **ground‑truth poses** (≈4 MB) from the [KITTI Odometry Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).  
2. In `main.py`, set:
   ```python
   groundtruth_file = "/path/to/poses.txt"
   image_folder     = "/path/to/image/sequences/"
3. Ensure that your folder structure matches KITTI’s format so that each frame aligns correctly with its ground‑truth pose.

### 📚 Pre-trained Model
We build on MagicLeap’s **SuperGluePretrainedNetwork** for feature matching:

1. Clone the official repo into your working directory:
```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
```
2. Verify that the `SuperGluePretrainedNetwork` folder sits alongside `main.py` so imports resolve cleanly.
3. The supplied weights (for both SuperPoint and SuperGlue) will be automatically downloaded on first run—no additional steps required.


### How to Run
```bash
python main.py
```
