#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=fovcnn
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=END
#SBATCH --output=/u/erdos/cnslab/xcui32/EfficientVideoRec/results/fovcnn/output.out

module purg
module load gcc8 cuda10.2
module load openmpi-geib-cuda10.2-gcc8/3.1.4
module load pytorch-py37-cuda10.2-gcc8/1.9.1
module load ml-pythondeps-py37-cuda10.2-gcc8/4.7.5


python3 /u/erdos/cnslab/xcui32/EfficientVideoRec/main.py \
 --root  \
 --start_epoch 0 --max_epoch 100 \
 --lr 0.001 --optimizer adam --lr_scheduler step --step_size 20 --gamma 0.2 \
 --momentum 0.9  --weight_decay 0.0005 --val_interval 1 \
 --num_workers 1 --batch_size 1  --device 'cuda:0' --download False --result_dir './results/fovcnn' --save False --resume --init_backbone
