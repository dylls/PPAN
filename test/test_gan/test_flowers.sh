#!/usr/bin/env bash

name='flowers_test_sampling10'
#name='test_sampling10'
CUDA_VISIBLE_DEVICES=${device} python3 test_worker.py \
                                    --dataset flowers \
                                    --model_name ${name} \
                                    --load_from_epoch 640 \
                                    --test_sample_num 26 \
                                    --save_visual_results \
                                    --batch_size 9
