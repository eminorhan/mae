#!/bin/bash

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=mae_finetune_imagenet
#SBATCH --output=mae_finetune_imagenet_%A_%a.out
#SBATCH --array=0

SUBJECTS=(
	"sayavakepicutego4d" 
	"sayavakepicutego4d_0.1_1" 
	"sayavakepicutego4d_0.1_2" 
	"sayavakepicutego4d_0.1_3" 
	"sayavakepicutego4d_0.01_1" 
	"sayavakepicutego4d_0.01_2" 
	"sayavakepicutego4d_0.01_3" 
	"sayavakepicutego4d_0.001_1" 
	"sayavakepicutego4d_0.001_2" 
	"sayavakepicutego4d_0.001_3" 
	"sayavakepicutego4d_0.0001_1" 
	"sayavakepicutego4d_0.0001_2" 
	"sayavakepicutego4d_0.0001_3"
	)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo $SUBJECT

# vith14 @ 476px
python -u /scratch/eo41/mae/eval_finetune.py \
	--model vit_huge_patch14_476 \
	--resume "/vast/eo41/sayavakepicutego4d_models/mae_vith14_476/${SUBJECT}_vith14_476_checkpoint.pth" \
	--save_prefix ${SUBJECT}_mae_vith14_476_1.0 \
	--input_size 476 \
	--batch_size 44 \
	--epochs 100 \
	--num_workers 16 \
	--output_dir "/scratch/eo41/regularized_humanlike_vits" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 1.0 \
	--num_labels 1000 \
	--no_optim_resume

echo "Done"
