#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=s_finetune_imagenet_h
#SBATCH --output=s_finetune_imagenet_h_%A_%a.out
#SBATCH --array=0

# # s_vith14
# python -u ../eval_finetune.py \
# 	--model vit_huge_patch14 \
# 	--resume ../models_vith14/s_vith14_checkpoint.pth \
# 	--save_prefix s_vith14_imagenet_0.02 \
# 	--batch_size 119 \
# 	--epochs 100 \
# 	--num_workers 16 \
# 	--lr 0.0001 \
# 	--output_dir ../models_vith14_imagenet \
# 	--train_data_path /scratch/work/public/imagenet/train \
# 	--val_data_path /scratch/eo41/imagenet/val \
# 	--frac_retained 0.02 \
# 	--num_labels 1000 \
# 	--no_optim_resume

# random_vith14
python -u ../eval_finetune.py \
	--model vit_huge_patch14 \
	--resume '' \
	--save_prefix random_vith14_imagenet_0.02 \
	--batch_size 119 \
	--epochs 100 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir ../models_vith14_imagenet \
	--train_data_path /scratch/work/public/imagenet/train \
	--val_data_path /scratch/eo41/imagenet/val \
	--frac_retained 0.02 \
	--num_labels 1000 \
	--no_optim_resume

echo "Done"
