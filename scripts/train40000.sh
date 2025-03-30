#!/bin/bash
# chmod +x scripts/train_lerf.sh
# ./scripts/train_lerf.sh

# !!! Please check the dataset path specified by -s.
# 假设你的 Anaconda 安装在默认路径
eval "$(conda shell.bash hook)"
conda activate opengs

# # ###############################################
scan="waldo_kitchen"
gpu_num=1        # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 10 \
    --min_cluster_size 400 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/kitchen_infonce_nearest/chkpnt40000.pth #/data/sunwei/gaussian-splatting/output/waldo_kitchen/chkpnt30000.pth #

scan="waldo_kitchen"
gpu_num=1        # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 10 \
    --min_cluster_size 800 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/kitchen_infonce_nearest/chkpnt40000.pth #/data/sunwei/gaussian-splatting/output/waldo_kitchen/chkpnt30000.pth #

scan="waldo_kitchen"
gpu_num=1        # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 10 \
    --min_cluster_size 1200 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/kitchen_infonce_nearest/chkpnt40000.pth #/data/sunwei/gaussian-splatting/output/waldo_kitchen/chkpnt30000.pth #

scan="waldo_kitchen"
gpu_num=1        # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 10 \
    --min_cluster_size 2000 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/kitchen_infonce_nearest/chkpnt40000.pth #/data/sunwei/gaussian-splatting/output/waldo_kitchen/chkpnt30000.pth #

