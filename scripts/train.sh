python train.py --dataroot ./deepfashion/fashion_resize --dirSem ./deepfashion --pairLst ./deepfashion/fashion-resize-pairs-train.csv --name fashion_adgan_test --model adgan --lambda_GAN 5 --lambda_A 1 --lambda_B 1 --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 6 --pool_size 0 --resize_or_crop no --gpu_ids 8,9 --BP_input_nc 18 --SP_input_nc 8 --no_flip --which_model_netG ADGen --niter 500 --niter_decay 500 --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1 --display_id 0