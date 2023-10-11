## Training MAEs with very large images

Some experiments to train a big MAE (ViT-H/14) with a small number of very high resolution images (currently: 1232x1232 pixels). The images are randomly sampled from the SA-1B dataset (canonical size: 2250x1500 pixels). 

### Usage example

* **Training:** We use [`train_mae_nowds.py`](https://github.com/eminorhan/mae/blob/master/train_mae_nowds.py) for these experiments: 
```python
python -u train_mae_nowds.py \
	--model 'mae_vit_huge_patch14' \
	--resume "" \
	--accum_iter 32 \
	--batch_size_per_gpu 1 \
	--input_size 1232 \
	--mask_ratio 0.8 \
	--num_workers 16 \
	--lr 0.00005 \
	--min_lr 0.00005 \
	--weight_decay 0.0 \
	--output_dir "/scratch/eo41/mae/giga_expt/models_1" \
	--data_path "/vast/eo41/sa-1b/images_1/0" \
	--save_prefix "giga_vith14_1_0"
```