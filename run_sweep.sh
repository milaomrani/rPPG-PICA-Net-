#!/bin/bash
set -e
OUT=../final-report/q5_assets
CACHE=../dataset/cache
ROOT=../dataset/UBFC_DATASET/DATASET_2
DEVICE=cpu

for seed in 42 123 7; do
  echo "=== no_augment seed=$seed (device=$DEVICE) ==="
  python3 -u run_pipeline.py --dataset_root $ROOT --cache_dir $CACHE --out_dir $OUT/no_augment_seed$seed \
    --epochs 60 --batch_size 4 --device $DEVICE --seed $seed
done

for seed in 42 123 7; do
  echo "=== with_augment seed=$seed (device=$DEVICE) ==="
  python3 -u run_pipeline.py --dataset_root $ROOT --cache_dir $CACHE --out_dir $OUT/with_augment_seed$seed \
    --epochs 60 --batch_size 4 --device $DEVICE --seed $seed --physics_augment
done

echo "SWEEP COMPLETE"
