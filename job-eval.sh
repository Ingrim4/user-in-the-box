#!/bin/bash
#SBATCH --job-name=uitb-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --time=00:10:00
#SBATCH -o logs/%x-%j/out.log
#SBATCH -e logs/%x-%j/err.log

module load Anaconda3
module load Mesa/23.1.9-GCCcore-13.2.0
export MUJOCO_GL=egl
export WANDB_MODE=disabled

source activate uitb
conda run -n uitb python uitb/test/evaluator.py simulators/mirror_09_001__x__35__abs__zero --num_episodes 1 --action_sample_freq 100 --record