#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_linear_imagenet
#SBATCH --output=eval_linear_imagenet_%A_%a.out
#SBATCH --array=0

NUM_IMGS=10000
MODEL_DIR=models_${NUM_IMGS}

# imagenet
srun python -u ../eval_linear.py \
	--model vit_huge_patch14 \
	--resume pretrained_models/${MODEL_DIR}/giga_vith14_m80_${NUM_IMGS}_${SLURM_ARRAY_TASK_ID}_checkpoint.pth \
	--save_prefix giga_vith14_m80_${NUM_IMGS}_${SLURM_ARRAY_TASK_ID}_eval_linear_imagenet \
	--batch_size 1024 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir eval_linear/${MODEL_DIR} \
	--train_data_path /scratch/work/public/imagenet/train \
	--val_data_path /scratch/eo41/imagenet/val \
	--num_labels 1000
	
echo "Done"
