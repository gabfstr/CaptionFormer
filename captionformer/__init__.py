# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer/__init__.py
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_captionformer_config

# dataset loading
from .data.dataset_mappers.coco_dense_dvoc_new_baseline_dataset_mapper import COCODenseDVOCNewBaselineDatasetMapper
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .captionformer_model_video import CaptionFormerVideo
from .captionformer_model import CaptionFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator


from .data_video import (
    YTVISDatasetMapper,
    YTVISDenseDVOCDatasetMapper,
    YTVISEvaluator,
    OVISEvaluator,
    LVVISEvaluator,
    LVVISEvaluator_video,
    BURSTEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
