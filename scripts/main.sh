#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=node002
#SBATCH --job-name=vit_voc10
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=END
#SBATCH --output=/u/erdos/cnslab/xcui32/EfficientVideoRec/results/vit_voc10/output.out

module purg
module load gcc8 cuda11.2
module load openmpi/cuda/64
module load ml-pythondeps-py37-cuda11.2-gcc8/4.7.8

source /u/erdos/cnslab/xcui32/venv/bin/activate

python3 /u/erdos/cnslab/xcui32/EfficientVideoRec/main.py \
 --root '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/' \
 --mode reducer-img --train True --patch_size 16 --reducer_inner_dim 32 --keep_ratio 0.8 \
 --backbone resnet18 --model vit_tiny_patch16_224 --reducer RandomReducer --backbone_out_dim 512 --pe --per_size --base_channels 64 \
 --start_epoch 0 --max_epoch 30 --object_only False --subset_data False \
 --lr 0.001 --optimizer adam --lr_scheduler step --step_size 20 --gamma 0.2 \
 --momentum 0.9  --weight_decay 0.0005 --val_interval 1 \
 --num_workers 16 --batch_size 64 --device 'cuda:0' --download False \
 --result_dir './results/vit_voc20' --save False --resume --init_backbone --pretrained True