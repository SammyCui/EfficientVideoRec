source /Users/xuanmingcui/Documents/cnslab/venv/bin/activate

python3 /Users/xuanmingcui/Documents/cnslab/EfficientVideoRec/main.py \
 --root '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered' \
 --start_epoch 0 --max_epoch 100 \
 --lr 0.001 --optimizer adam --lr_scheduler step --step_size 20 --gamma 0.2 \
 --momentum 0.9  --weight_decay 0.0005 --val_interval 1 \
 --num_workers 1 --batch_size 1 --device 'cpu' --download False --result_dir './results/fovcnn' --save False --resume --init_backbone