#!/usr/bin/env bash
## test 512 resolution
#name='birds_512'
#CUDA_VISIBLE_DEVICES=${device} python test_worker.py \
#                                    --dataset birds \
#                                    --model_name ${name} \
#                                    --load_from_epoch 80 \
#                                    --test_sample_num 10 \
#                                    --finest_size 512 \
#                                    --batch_size 4

# test 256
name='add64_128_256class_loss_PPAN_256_birds'
CUDA_VISIBLE_DEVICES=${device} python3 test_worker.py \
                                    --dataset birds \
                                    --model_name ${name} \
                                    --load_from_epoch 520 \
                                    --test_sample_num 10 \
                                    --save_visual_results \
                                    --batch_size 9


