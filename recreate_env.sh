#! /bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda deactivate
conda remove --name fake-news-gpu --all -y
conda env create -f environment_gpu.yml