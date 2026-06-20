# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer_video/__init__.py
from . import modeling

# config
from .config import add_captionformer_video_config

# models
from .video_captionformer_model import VideoCaptionFormer

