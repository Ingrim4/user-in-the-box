#!/bin/bash
#SBATCH --job-name=uitb-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --time=48:00:00
#SBATCH -o logs/%x-%j/out.%a.log
#SBATCH -e logs/%x-%j/err.%a.log

module load Anaconda3
module load Mesa/23.1.9-GCCcore-13.2.0
export MUJOCO_GL=egl
export WANDB_MODE=disabled

source activate uitb
conda run -n uitb python uitb/train/trainer.py uitb/configs/mobl_arms_index_mirroring_test.yaml