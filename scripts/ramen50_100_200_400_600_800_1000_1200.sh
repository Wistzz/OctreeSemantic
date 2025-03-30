eval "$(conda shell.bash hook)"
conda activate opengs

min_cluster_sizes=(50 100 200 400 600 800 1000 1200)
scan="ramen"
gpu_num=2

for min_cluster_size in "${min_cluster_sizes[@]}"; do
    echo "Training for ${scan} ....."
    CUDA_VISIBLE_DEVICES=$gpu_num python train40000.py --port 421$gpu_num \
        -s /data/sunwei/OpenGaussian/data/lerf_ovs/${scan} \
        --iterations 40001 \
        --start_ins_feat_iter 30000 \
        --start_root_cb_iter 40000 \
        --start_leaf_cb_iter 50000 \
        --sam_level 3 \
        --cluster_num 256 \
        --pos_weight 0.1 \
        --min_cluster_size $min_cluster_size \
        --save_memory \
        --test_iterations 30000 \
        --spatial_grid_size 2.0 \
        --eval \
        --start_checkpoint /data/sunwei/OctreeSemantic/output/ramen_infonce_nearest/chkpnt40000.pth
done
    