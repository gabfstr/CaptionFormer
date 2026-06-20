# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer/config.py
from detectron2.config import CfgNode as CN


def add_captionformer_config(cfg):
    """
    Add config for captionformer.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    
    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()


    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.OBJECT_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    
    cfg.MODEL.MASK_FORMER.USE_MASKS = True
    cfg.MODEL.MASK_FORMER.USE_BOXES = False
    # For Vidstg you need box instead of mask
    cfg.MODEL.MASK_FORMER.BOX_ONLY = False
    cfg.MODEL.MASK_FORMER.BOX_MODE_ON = False
    cfg.MODEL.MASK_FORMER.TEMPORAL_POS_EMBEDDING = False
    cfg.MODEL.MASK_FORMER.DISABLE_ATTN_MASK = False
    cfg.MODEL.MASK_FORMER.EMPTY_OBJECT_WEIGHT = 0.4
    cfg.MODEL.MASK_FORMER.BOX_XYXY = False
    # Losses when box mode on 
    cfg.MODEL.MASK_FORMER.BOX_LOSS_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.GIOU_LOSS_WEIGHT = 2.0


    ##### Experimental
    cfg.MODEL.MASK_FORMER.MIDDLE_FRAME_CAPTIONING = False
    cfg.MODEL.MASK_FORMER.NUM_CAPTIONS_PER_VIDEO = 1
    cfg.MODEL.MASK_FORMER.VIDEO_LEVEL_TRAINING = False
    cfg.MODEL.MASK_FORMER.DEBUGGING_MODE_ON = False
    cfg.MODEL.MASK_FORMER.FILTER_PERSON_ONLY = False
    ##### Inference
    cfg.MODEL.MASK_FORMER.DVOC_INFERENCE = False
    cfg.MODEL.MASK_FORMER.DVOC_INFERENCE_THRESHOLD = 0.5
    cfg.MODEL.MASK_FORMER.DVOC_CLASS_AGNOSTIC = False
    cfg.MODEL.MASK_FORMER.CLASS_AGNOSTIC_INFERENCE = False
    cfg.MODEL.MASK_FORMER.DUMMY_CLASS_PREDICTION = False
    cfg.MODEL.MASK_FORMER.LVVIS_ZS_VIDSTG_CLASS_FILTERING = False

    # Hysteresis filtering
    cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER = CN()
    cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.ENABLED = False
    cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.T_LOW = 0.5
    cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.T_HIGH = 0.5
    cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.MIN_DURATION = 3
    # Per-frame score threshold
    cfg.MODEL.MASK_FORMER.PER_FRAME_SCORE_THRESHOLD = 0.0

    ##### Tracking
    cfg.MODEL.MASK_FORMER.HUNGARIAN_TRACK_MATCHING = False
    cfg.MODEL.MASK_FORMER.TRACKING_EMBED_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.TRACKING_OBJECT_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.TRACKING_CLASS_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.TRACKING_MASK_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.TRACKING_DICE_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.TRACKING_BOX_L1_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.TRACKING_BOX_GIOU_WEIGHT = 0.0

    cfg.MODEL.MASK_FORMER.TOPK_MATCHING = False
    cfg.MODEL.MASK_FORMER.TOPK_MATCHING_NUM_K = 3
    cfg.MODEL.MASK_FORMER.TOPK_MATCHING_NUM_T = 9

    cfg.MODEL.MASK_FORMER.GREEDY_MATCHING=False


    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32


    cfg.MODEL.MASK_FORMER.CLIP_TEXT_PATH = ''
    cfg.MODEL.MASK_FORMER.CLIP_IMAGE_PATH = ''

    # classifier config
    cfg.MODEL.MASK_FORMER.CLIP_CLASSIFIER = False
    cfg.MODEL.MASK_FORMER.AGNOSTIC_CLASSIFIER = False

    #CLIP LEN 
    cfg.MODEL.MASK_FORMER.INFERENCE_CLIP_LEN = 5

    # Mask captioning
    cfg.MODEL.MASK_FORMER.MASK_CAPTIONING = False
    cfg.MODEL.MASK_FORMER.DUMMY_CAPTIONING = False
    cfg.MODEL.MASK_FORMER.TUNE_CAPTIONING_HEAD = False
    cfg.MODEL.MASK_FORMER.CAPTION_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.CAPTIONING_HEAD_NAME = "Blip2OPT"
    cfg.MODEL.MASK_FORMER.CAPTIONING_THRESH = 0.3
    cfg.MODEL.MASK_FORMER.CAPTIONING_MAX_OUT_LEN = 30
    cfg.MODEL.MASK_FORMER.CAPTIONING_REPETITION_PENALTY = 1.0
    cfg.MODEL.MASK_FORMER.MULTI_FRAME_CAPTIONING = False
    cfg.MODEL.MASK_FORMER.TEST_AGG_MIDDLE_CLIP = False

    # cfg.MODEL.MASK_FORMER.CONCAT_AGGREGATION = False
    # cfg.MODEL.MASK_FORMER.MEAN_AGGREGATION = False
    # cfg.MODEL.MASK_FORMER.AGGREGATION_NUM_T = 4

    # Captioning head
    cfg.MODEL.CAPTIONING_HEAD = CN()
    cfg.MODEL.CAPTIONING_HEAD.VIT_MODEL_NAME = "eva_clip_g"
    cfg.MODEL.CAPTIONING_HEAD.SAVED_FEATURES = True
    cfg.MODEL.CAPTIONING_HEAD.FEATURE_MAPPING = "lvvis"
    cfg.MODEL.CAPTIONING_HEAD.IMG_SIZE = 364
    cfg.MODEL.CAPTIONING_HEAD.NUM_TEXT_QUERIES = 32
    cfg.MODEL.CAPTIONING_HEAD.OPT_MODEL = ""
    cfg.MODEL.CAPTIONING_HEAD.DROP_PATH_RATE = 0.0
    cfg.MODEL.CAPTIONING_HEAD.USE_GRAD_CHECKPOINT = False
    cfg.MODEL.CAPTIONING_HEAD.VIT_PRECISION = "fp32"
    cfg.MODEL.CAPTIONING_HEAD.FREEZE_VIT = False
    cfg.MODEL.CAPTIONING_HEAD.PROMPT = ""
    cfg.MODEL.CAPTIONING_HEAD.MAX_TXT_LEN = 32
    cfg.MODEL.CAPTIONING_HEAD.APPLY_LEMMATIZER = False   
    # video level captioning
    cfg.MODEL.CAPTIONING_HEAD.AGGREGATION_METHOD = "concat"  # "mean", "concat"
    cfg.MODEL.CAPTIONING_HEAD.OPT_PROJ_BEFORE_AGGREGATION = False
    cfg.MODEL.CAPTIONING_HEAD.AGGREGATION_NUM_T = 4
    # Inference
    cfg.MODEL.CAPTIONING_HEAD.USE_NUCLEUS_SAMPLING = False
    cfg.MODEL.CAPTIONING_HEAD.NUM_BEAMS = 5
    cfg.MODEL.CAPTIONING_HEAD.TOP_P = 0.9
    cfg.MODEL.CAPTIONING_HEAD.TEMPERATURE = 1.0

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    cfg.MODEL.TIMM = CN()
    cfg.MODEL.TIMM.BASE_NAME = 'resnet50'
    cfg.MODEL.TIMM.OUT_LEVELS = (2, 3, 4, 5)
    cfg.MODEL.TIMM.NORM = 'FrozenBN'
    cfg.MODEL.TIMM.FREEZE_AT = 0
    cfg.MODEL.TIMM.PRETRAINED = False

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.SAMPLING_FULL_VIDEO = False

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
