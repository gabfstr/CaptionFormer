#!/bin/bash
# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

# CaptionFormer eval on VLN with temporal aggregation, mask mode 

EVAL_OUT=./outputs/eval/vln

python train_net_lvvis.py --num-gpus 1 --eval-only \
    --config-file configs/eval/vln.yaml \
    OUTPUT_DIR "$EVAL_OUT/"

python evaluate_chota_full.py \
    --dt_json "$EVAL_OUT/inference/results.json" \
    --gt_json ./datasets/VLN/annotations/vng_uvo_sparse_val_instances_coco_format.json \
    --vln_extended_gt ./datasets/VLN/annotations/vng_uvo_sparse_val_extended_instances_.json \
    -masks -cider_only
