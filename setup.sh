#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda env export > environment.yaml" after running this.
#
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

set -e

conda create -n BCI python=3.10.8 -y
conda activate BCI

conda install numpy matplotlib tqdm scikit-image jupyterlab -y
conda install -c conda-forge accelerate -y

pip install clip-retrieval clip pandas matplotlib ftfy regex kornia umap-learn
pip install dalle2-pytorch

pip install open_clip_torch

pip install transformers
pip install diffusers

pip install braindecode

pip install torchvision torch

pip install info-nce-pytorch
pip install pytorch-msssim

pip install reformer_pytorch

pip install mne
pip install wandb
pip install einops