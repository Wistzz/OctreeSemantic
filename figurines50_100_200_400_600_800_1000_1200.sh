eval "$(conda shell.bash hook)"
conda activate opengs

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 50 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 100 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 200 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 400 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 600 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 800 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 800 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 1000 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth

scan="figurines"
gpu_num=0
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
    -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 40001 \
    --start_ins_feat_iter 30000 \
    --start_root_cb_iter 40000 \
    --start_leaf_cb_iter 50000 \
    --sam_level 3 \
    --cluster_num 256 \
    --pos_weight 0.5 \
    --min_cluster_size 1200 \
    --save_memory \
    --test_iterations 30000 \
    --spatial_grid_size 2.0 \
    --eval \
    --start_checkpoint /data/sunwei/OctreeSemantic/output/figurines_infonce_nearest/chkpnt40000.pth