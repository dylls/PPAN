#!/usr/bin/env bash

name='flowers_add64_128_256class_loss_PPAN_256'
dataset='flowers'
dir='/home/* /PPAN/Models/'${name}_$dataset
mkdir -v $dir
CUDA_VISIBLE_DEVICES=${device} python3 train_worker.py \
                                --dataset $dataset \
                                --batch_size 7 \
                                --model_name ${name} \
                                --g_lr 0.0002 \
                                --d_lr 0.0002 \
                                | tee $dir/'log.txt'