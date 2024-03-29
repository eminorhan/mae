#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=00:15:00
#SBATCH --job-name=eval_video_seg_score
#SBATCH --output=eval_video_seg_score_%A_%a.out
#SBATCH --array=0-11

module purge
module load cuda/11.6.2

MODELS=(vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(say s a y say s a y say s a y)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT

srun python -u /scratch/eo41/davis2017-evaluation/evaluation_method.py \
	--task semi-supervised \
	--results_path "/scratch/eo41/mae/evals/davis-2017/${SUBJECT}_${MODEL}" \
	--davis_path "/vast/eo41/data/davis-2017/DAVIS"

echo "Done"