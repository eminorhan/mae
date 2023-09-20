#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=00:10:00
#SBATCH --job-name=train_giga_mae
#SBATCH --output=train_giga_mae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# vit-s/15
srun python -u /scratch/eo41/mae/train_mae_nowds.py \
	--model 'mae_vit_small_patch14' \
	--resume "" \
	--batch_size_per_gpu 1 \
	--input_size 1400 \
	--mask_ratio 0.8 \
	--num_workers 16 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--weight_decay 0.0 \
	--output_dir "/scratch/eo41/mae/giga_expt" \
	--data_path "/vast/eo41/sa-1b/test" \
	--save_prefix "giga_vits15"

# # vit-b/15
# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_base_patch14' \
# 	--resume "" \
# 	--batch_size_per_gpu 256 \
#	--mask_ratio 0.8 \
# 	--num_workers 16 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/vast/eo41/sayavakepicutego4d_models/mae_vitb14" \
# 	--data_path "/vast/eo41/sayavakepicutego4d/${DATA}.tar" \
# 	--save_prefix "${SAVE}_vitb14"

echo "Done"