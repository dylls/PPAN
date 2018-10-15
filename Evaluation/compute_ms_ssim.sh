# bird
CUDA_VISIBLE_DEVICES=${device}  python3 ms_ssim/msssim_score.py \
                                --image_folder /home/*/raw_PPAN/Results/birds/PPAN_256_birds_testing_num_10 \
                                --h5_file 'PPAN_256_birds_G_epoch_595.h5' \
                                --evaluate_overall_score
