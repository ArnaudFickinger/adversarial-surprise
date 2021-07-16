#!/bin/bash
conda env create -f conda_env.yml
source activate adversarial_surprise
conda install pytorch torchvision torchaudio -c pytorch
python run_minigrid.py