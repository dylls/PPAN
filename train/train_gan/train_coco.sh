#!/usr/bin/env bash

name='coco_add64_128_256class_loss_PPAN_256'
dataset='coco'
dir='/home/*/PPAN/Models/'${name}_$dataset
mkdir -v $dir
CUDA_VISIBLE_DEVICES=${device} python3 train_worker.py \
                                --dataset $dataset \
                                --batch_size 6 \
                                --model_name ${name} \
                                --g_lr 0.0002 \
                                --d_lr 0.0002 \
                                --epoch_decay 50 \
                                --KL_COE 2 \
                                --gpus ${device} \
                                | tee $dir/'log.txt'

# need about 150 epochs# original 0.0000001

