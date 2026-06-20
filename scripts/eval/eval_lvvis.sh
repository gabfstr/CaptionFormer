#!/bin/bash
# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

# CaptionFormer eval on LVVIS captioner (paper table 2 ablation)

EVAL_OUT=./outputs/eval/lvvis_captioner

python train_net_lvvis.py --num-gpus 1 --eval-only \
    --config-file configs/eval/lvvis_captioner.yaml \
    OUTPUT_DIR "$EVAL_OUT/"

python evaluate_chota_full.py \
    --dt_json "$EVAL_OUT/inference/results.json" \
    --gt_json ./datasets/LVVIS/lvviscap_val_instances_coco_format.json \
    -masks -cider_only
