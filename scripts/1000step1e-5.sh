eval "$(conda shell.bash hook)"
conda activate opengs

# ============== [10 scenes] ==============
scan_list=("scene0000_00" "scene0062_00" "scene0070_00" "scene0097_00" "scene0140_00" \
"scene0200_00" "scene0347_00" "scene0400_00" "scene0590_00" "scene0645_00")
gpu_num=2     # change!
for scan in "${scan_list[@]}"; do
    echo "Training for ${scan} ....."
    CUDA_VISIBLE_DEVICES=$gpu_num python train_1e-5.py --port 201$gpu_num \
        -s /data/sunwei/OpenGaussian/data/${scan} \
        -r 2 \
        --frozen_init_pts \
        --iterations 50_000 \
        --start_ins_feat_iter 30_000 \
        --start_root_cb_iter 50_000 \
        --start_leaf_cb_iter 70_000 \
        --sam_level 0 \
        --temp 0.1 \
        --min_cluster_size 10 \
        --pos_weight 0 \
        --test_iterations 30000 \
        --eval \
        # --start_checkpoint /data/sunwei/OctreeSemantic/output/scene0062_00/chkpnt30000.pth #/data/sunwei/OctreeSemantic/output/scene0000_00/chkpnt50000.pth #/data/sunwei/OctreeSemantic/output/scene0000_00/chkpnt30000.pth 
done