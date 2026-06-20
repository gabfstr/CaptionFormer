#!/bin/bash
# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

# CaptionFormer eval on BenSMOT with temporal aggregation 

EVAL_OUT=./outputs/eval/bensmot
GT=./datasets/bensmot/annotations/test_instances_coco_format.json

# Step 1 — model inference
python train_net_lvvis.py --num-gpus 1 --eval-only \
    --config-file configs/eval/bensmot.yaml \
    OUTPUT_DIR "$EVAL_OUT/"

# Step 2 — HOTA scores (DetA, AssA, etc.)
python evaluate_chota_full.py \
    --dt_json "$EVAL_OUT/inference/results.json" \
    --gt_json $GT \
    -no_caption

# Step 3 — BenSMOT-paper captioning metrics (plain CIDEr/BLEU/METEOR/ROUGE).
# BenSMOT reports raw CIDEr, NOT CapA-CIDEr (per-track mean box-IoU , from BenSMOT paper)
python evaluate/smot_utils/results2smot_cap.py \
    --gt_json $GT \
    --dt_json "$EVAL_OUT/inference/results.json"
# results2smot_cap.py writes to <dt_json_dir>/pred_results/caption_results.json
python evaluate/smot_utils/eval_vu.py \
    --res_folder "$EVAL_OUT/inference/pred_results/" \
    -caption_only
