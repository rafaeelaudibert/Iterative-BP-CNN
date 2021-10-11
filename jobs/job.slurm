#!/bin/bash

#SBATCH --job-name=iterative-bp-cnn
#SBATCH -o output.txt -e error.txt
#SBATCH --mem=16000 # Allocate 16 GB of RAM
#SBATCH --gres=gpu:K80:1 # Choose GPU i.e. K20Xm, K80, V100
#SBATCH --time=96:00:00 # Choose time limit of job in format hh:mm:ss

module load nvidia/latest
module load anaconda3/latest

. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate iterative-bp-cnn

python3 ~/Iterative-BP-CNN/main.py -Func Train

conda deactivate
