#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=12:10:00
#SBATCH --job-name=eval_finetune_imagenet
#SBATCH --output=eval_finetune_imagenet_%A_%a.out
#SBATCH --array=0

NUM_IMGS=10000
MODEL_DIR=models_${NUM_IMGS}

srun python -u ../eval_finetune_accum.py \
	--model vit_huge_patch14_896 \
	--resume pretrained_models/${MODEL_DIR}/giga_vith14_m80_${NUM_IMGS}_${SLURM_ARRAY_TASK_ID}_checkpoint.pth \
	--save_prefix giga_vith14_m80_${NUM_IMGS}_${SLURM_ARRAY_TASK_ID}_eval_finetune_imagenet \
	--input_size 896 \
	--batch_size 1 \
	--accum_iter 128 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir eval_finetune/${MODEL_DIR} \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 0.01 \
	--num_labels 1000 \
	--no_optim_resume

echo "Done"
