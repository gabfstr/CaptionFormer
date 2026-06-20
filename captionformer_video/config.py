# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer_video/config.py
from detectron2.config import CfgNode as CN


def add_captionformer_video_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
