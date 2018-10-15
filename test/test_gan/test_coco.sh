#!/usr/bin/env bash

name='coco_add64_128_256class_loss_PPAN_256_coco'
CUDA_VISIBLE_DEVICES=${device} python3 test_worker.py \
                                --dataset coco \
                                --model_name ${name} \
                                --load_from_epoch 55 \
                                --test_sample_num 1 \
                                --batch_size 8 \
                                --save_visual_results \

