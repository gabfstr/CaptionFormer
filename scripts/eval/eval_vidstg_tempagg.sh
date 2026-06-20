#!/bin/bash
# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

# CaptionFormer eval on VidSTG with temporal aggregation

EVAL_OUT=./outputs/eval/vidstg_tempagg

python train_net_lvvis.py --num-gpus 1 --eval-only \
    --config-file configs/eval/vidstg_tempagg.yaml \
    OUTPUT_DIR "$EVAL_OUT/"

python evaluate_chota_full.py \
    --dt_json "$EVAL_OUT/inference/results.json" \
    --gt_json ./datasets/VidSTG/annotations/vidstg_max200f_val_coco_format.json \
    -cider_only
