#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=node002
#SBATCH --job-name=fovcnn
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=END
#SBATCH --output=/u/erdos/cnslab/xcui32/EfficientVideoRec/results/fovcnn/output.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc

python3 /u/erdos/cnslab/xcui32/EfficientVideoRec/main.py \
 --root '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/' \
 --start_epoch 0 --max_epoch 100 \
 --lr 0.001 --optimizer adam --lr_scheduler step --step_size 20 --gamma 0.2 \
 --momentum 0.9  --weight_decay 0.0005 --val_interval 1 \
 --num_workers 1 --batch_size 1 --device 'cuda:0' --download False --result_dir './results/fovcnn' --save False --resume --init_backbone
