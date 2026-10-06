#!/bin/bash
# 闸门实验 (交接文档 7.2 节命令, 缓存特征版): CUB 4 组 × 3 seeds
# 用法: bash run_gate_cub.sh [dataset]   # 默认 CUB, 可传 AWA2
set -u
PY=/d/anaconda3/envs/trl4060/python.exe
DS=${1:-CUB}
CLIP="F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
cd /f/CLIP-OOD

COMMON="--dataset $DS --epochs 100 --batch_size 256 --alpha 1.0 --beta 0 --CLIP_type $CLIP"

run () {  # run <tag> <extra args...>
  local tag=$1; shift
  for seed in 1 2 3; do
    echo "===== [$DS] $tag seed=$seed  $(date +%H:%M:%S) ====="
    $PY main_cached.py $COMMON --seed $seed "$@" > "downloads/gate_${DS}_${tag}_s${seed}.log" 2>&1
    grep -a "Best Target Accuracy" "downloads/gate_${DS}_${tag}_s${seed}.log" | tail -1
  done
}

run ddo     --CBM_type clip_cbm --weight_mode none
run c3      --CBM_type clip_cbm --weight_mode prior_residual --lambda_kl 0.01
run c4      --CBM_type clip_cbm_subspace --ch4_mode full --tau 0.8 --lambda_inv 1.0
run c3+c4   --CBM_type clip_cbm_subspace --ch4_mode full --tau 0.8 --lambda_inv 1.0 \
            --use_c3_weights --weight_mode prior_residual --lambda_kl 0.01

echo "===== 全部完成 $(date) ====="
$PY aggregate_results.py --dataset $DS 2>/dev/null || $PY aggregate_results.py
