#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_mae
#SBATCH --output=train_mae_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_small_patch16' \
# 	--resume "/scratch/eo41/mae/models_vits16/say_5fps_vits16_checkpoint.pth" \
# 	--batch_size_per_gpu 512 \
# 	--num_workers 8 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/scratch/eo41/mae/models_vits16" \
# 	--data_path "/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar" \
# 	--save_prefix "say_5fps_vits16"

# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_small_patch16' \
# 	--resume "/scratch/eo41/mae/models_vits16/s_5fps_vits16_checkpoint.pth" \
# 	--batch_size_per_gpu 512 \
# 	--num_workers 8 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/scratch/eo41/mae/models_vits16" \
# 	--data_path "/scratch/eo41/data/saycam/S_5fps_300s_{000000..000003}.tar" \
# 	--save_prefix "s_5fps_vits16"

# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_small_patch16' \
# 	--resume "/scratch/eo41/mae/models_vits16/a_5fps_vits16_checkpoint.pth" \
# 	--batch_size_per_gpu 512 \
# 	--num_workers 8 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/scratch/eo41/mae/models_vits16" \
# 	--data_path "/scratch/eo41/data/saycam/A_5fps_300s_{000000..000002}.tar" \
# 	--save_prefix "a_5fps_vits16"
	
# srun python -u /scratch/eo41/mae/train_mae.py \
# 	--model 'mae_vit_small_patch16' \
# 	--resume "/scratch/eo41/mae/models_vits16/y_5fps_vits16_checkpoint.pth" \
# 	--batch_size_per_gpu 512 \
# 	--num_workers 8 \
# 	--lr 0.0003 \
# 	--min_lr 0.0003 \
# 	--weight_decay 0.0 \
# 	--output_dir "/scratch/eo41/mae/models_vits16" \
# 	--data_path "/scratch/eo41/data/saycam/Y_5fps_300s_{000000..000002}.tar" \
# 	--save_prefix "y_5fps_vits16"

echo "Done"