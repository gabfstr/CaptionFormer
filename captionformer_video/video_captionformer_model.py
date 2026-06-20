# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer_video/video_ovformer_model.py
from collections import deque
import logging
import math
from typing import Tuple
import os
import time
import json

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher, VideoHungarianTemporalMatcher
from .modeling.captioning_head import build_video_captioning_head

from .utils.memory import retry_if_cuda_oom

from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoCaptionFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        mask_captioning: bool,
        dummy_captioning: bool,
        captioning_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames: int,
        inference_clip_len: int,
        clip_classifier: bool,
        clip_image_path: str,
        # Box instead of mask 
        box_mode_on: bool,
        box_xyxy: bool,

        # freeze all but captioning head
        tune_captioning_head: bool,
        captioning_thresh: float,
        captioning_max_out_len: int,
        captioning_rep_penalty: float,
        captioning_feat_aggregation: bool,
        caption_num_t: int,
        test_agg_middle_clip: bool,
        
        # Experimental
        debugging_mode_on: bool,
        middle_frame_captioning: bool = False,
        num_captions_per_video: int = 1,
        video_level_training: bool = False,
        video_matcher: nn.Module = None,
        filter_person_only: bool = False,

        #dvoc inference
        dvoc_inference: bool,
        dvoc_threshold: float,
        dvoc_class_agnostic: bool,
        class_agnostic_inference: bool,
        dummy_class_prediction: bool,
        lvvis_zs_vidstg_class_filtering: bool,
        hysteresis_filter_enabled: bool,
        hysteresis_t_low: float,
        hysteresis_t_high: float,
        hysteresis_min_duration: int,
        per_frame_score_threshold: float,

        #tracking
        hungarian_track_matching_on: bool,
        hungarian_track_matcher: nn.Module,

        topk_matching: bool,
        topk_matching_num_k: int,
        topk_matching_num_t: int,

        greedy_matching: bool,

        #captioning inference parameters
        use_nucleus_sampling: bool = False,
        num_beams:int = 5,
        top_p: float = 0.9,
        temperature:float = 1.0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        ### 
        # Experimental
        self.debugging_mode_on = debugging_mode_on
        self.video_level_training = video_level_training
        if self.video_level_training:
            self.video_matcher = video_matcher
        self.filter_person_only = filter_person_only

        #dvoc inference
        self.dvoc_inference = dvoc_inference
        self.dvoc_threshold = dvoc_threshold
        self.dvoc_class_agnostic = dvoc_class_agnostic
        self.class_agnostic_inference = class_agnostic_inference
        self.dummy_class_prediction = dummy_class_prediction
        self.lvvis_zs_vidstg_class_filtering = lvvis_zs_vidstg_class_filtering
        self.hysteresis_filter_enabled = hysteresis_filter_enabled
        self.hysteresis_t_low = hysteresis_t_low
        self.hysteresis_t_high = hysteresis_t_high
        self.hysteresis_min_duration = hysteresis_min_duration
        self.per_frame_score_threshold = per_frame_score_threshold
        
        # Load LVVIS to VidSTG category mapping if filtering is enabled
        if self.lvvis_zs_vidstg_class_filtering:
            import json
            mapping_path = "./datasets/metadata/lvvis_to_vidstg_cat_mapping.json"
            with open(mapping_path, "r") as f:
                self.lvvis_to_vidstg_mapping = json.load(f)
            # Convert string keys to int and adjust for 0-based indexing
            # Mapping dict has IDs from 1-1196, but predictions are 0-based
            self.lvvis_to_vidstg_mapping = {int(k)-1: int(v)-1 for k, v in self.lvvis_to_vidstg_mapping.items()}
            print(f"Loaded LVVIS to VidSTG mapping: {len(self.lvvis_to_vidstg_mapping)} categories")
        else:
            self.lvvis_to_vidstg_mapping = None

        
        # tracking
        self.hungarian_track_matching_on = hungarian_track_matching_on
        self.hungarian_track_matcher = hungarian_track_matcher

        self.topk_matching = topk_matching
        self.topk_matching_num_k = topk_matching_num_k
        self.topk_matching_num_t = topk_matching_num_t

        self.greedy_matching = greedy_matching
        if self.greedy_matching:
            assert self.topk_matching == False, "Tracking cannot be both topk and greedy (offline)"

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        
        self.mask_captioning = mask_captioning
        self.dummy_captioning = dummy_captioning
        if self.dummy_captioning :
            self.category_names = MetadataCatalog.get(metadata.name).thing_classes
        self.middle_frame_captioning = middle_frame_captioning
        self.num_captions_per_video = num_captions_per_video
        self.captioning_thresh = captioning_thresh
        self.captioning_max_out_len = captioning_max_out_len
        self.captioning_rep_penalty = captioning_rep_penalty
        self.captioning_feat_aggregation = captioning_feat_aggregation
        self.caption_num_t = caption_num_t
        self.test_agg_middle_clip = test_agg_middle_clip
        if self.mask_captioning:
            self.captioning_head = captioning_head
        self.tune_captioning_head = tune_captioning_head

        # captioning inference parameters
        self.captioning_use_nucleus_sampling = use_nucleus_sampling
        self.captioning_num_beams = num_beams
        self.captioning_top_p = top_p
        self.captioning_temperature = temperature

        self.box_mode_on = box_mode_on
        self.box_xyxy = box_xyxy

        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        if self.mask_captioning:
            import os
            url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt2.7b.pth"
            path = os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "blip2_pretrained_opt2.7b.pth",
            )
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                print(f"downloading BLIP-2 pretrained weights to {path}")
                torch.hub.download_url_to_file(url, path, progress=True)
            weights = torch.load(path, map_location=self.device)
            # Change key from "language_model" to "opt_model"
            for k in list(weights.keys()):
                if "language_model" in k:
                    weights[k.replace("language_model", "opt_model")] = weights.pop(k)
                if "qformer" in k:
                    weights[k.replace("qformer", "Qformer.bert")] = weights.pop(k)
                if "vision_model.encoder.layers" in k:
                    weights[k.replace("vision_model.encoder.layers", "visual_encoder.encoder.blocks")] = weights.pop(k)
                if "vision_model" in k:
                    weights[k.replace("vision_model", "visual_encoder")] = weights.pop(k)
            
            print("loading BLIP2 weights")
            # self.captioning_head.load_state_dict(weights, strict=True)
            self.captioning_head.load_state_dict(weights, strict=False)

            print("Loaded captioning weights from:", path)

        self.num_frames = num_frames
        self.inference_clip_len = inference_clip_len
        
        self.clip_classifier = clip_classifier
        if self.clip_classifier:
            self.clip = torch.load(clip_image_path, map_location=self.device)
        
        self.i_debug=0
        if self.debugging_mode_on:
            self.i_debug=0
        
        # Timing statistics for inference
        self.timing_stats = {
            'detection_times': [],  # ms per video
            'tracking_times': [],   # ms per video
            'captioning_times': [], # ms per video
            'post_processing_times': [], # ms per video
            'total_times': [],      # ms per video
            'num_frames_per_video': [],  # for FPS calculation
            'per_video_stats': [],  # detailed stats per video
        }
        self.enable_timing = False  # Set to True during inference
        self.timing_video_count = 0  # Counter for videos processed

        # Freeze all weights but captioning head
        if self.tune_captioning_head:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.sem_seg_head.parameters():
                param.requires_grad = False
            # Leave captioning head as is (not frozen)

        
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        mask_captioning = cfg.MODEL.MASK_FORMER.MASK_CAPTIONING
        if cfg.MODEL.MASK_FORMER.MASK_CAPTIONING:
            captioning_head = build_video_captioning_head(cfg, cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        else : 
            captioning_head = None
        
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        object_weight = cfg.MODEL.MASK_FORMER.OBJECT_WEIGHT
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        caption_weight = cfg.MODEL.MASK_FORMER.CAPTION_WEIGHT
        box_l1_weight = cfg.MODEL.MASK_FORMER.BOX_LOSS_WEIGHT
        box_giou_weight = cfg.MODEL.MASK_FORMER.GIOU_LOSS_WEIGHT

        box_loss = cfg.MODEL.MASK_FORMER.BOX_MODE_ON

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_object=object_weight,
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_bbox=box_l1_weight,
            cost_giou=box_giou_weight,
            by = 'mask' if not box_loss else 'box',
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        if cfg.MODEL.MASK_FORMER.HUNGARIAN_TRACK_MATCHING:
            track_embed_weight = cfg.MODEL.MASK_FORMER.TRACKING_EMBED_WEIGHT
            track_object_weight = cfg.MODEL.MASK_FORMER.TRACKING_OBJECT_WEIGHT
            track_class_weight = cfg.MODEL.MASK_FORMER.TRACKING_CLASS_WEIGHT
            track_mask_weight = cfg.MODEL.MASK_FORMER.TRACKING_MASK_WEIGHT
            track_dice_weight = cfg.MODEL.MASK_FORMER.TRACKING_DICE_WEIGHT
            track_box_l1_weight = cfg.MODEL.MASK_FORMER.TRACKING_BOX_L1_WEIGHT
            track_box_giou_weight = cfg.MODEL.MASK_FORMER.TRACKING_BOX_GIOU_WEIGHT
            
            track_matcher= VideoHungarianTemporalMatcher(
                cost_embed=track_embed_weight,
                cost_object=track_object_weight,
                cost_class=track_class_weight,
                cost_mask=track_mask_weight,
                cost_dice=track_dice_weight,
                cost_bbox=track_box_l1_weight,
                cost_giou=track_box_giou_weight,
                by='box' if box_loss else 'mask',
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
        else : 
            track_matcher = None



        weight_dict = {"loss_object_ce": object_weight, "loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, 
                       "loss_caption": caption_weight, "loss_l1_box": box_l1_weight, "loss_giou_box": box_giou_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # Caption loss is NOT in criterion
        if box_loss:
            losses = ["labels", "boxes"]
        else :
            losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "mask_captioning": mask_captioning,
            "dummy_captioning": cfg.MODEL.MASK_FORMER.DUMMY_CAPTIONING,
            "captioning_head": captioning_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "inference_clip_len": cfg.MODEL.MASK_FORMER.INFERENCE_CLIP_LEN,
            "clip_classifier": cfg.MODEL.MASK_FORMER.CLIP_CLASSIFIER,
            "clip_image_path": cfg.MODEL.MASK_FORMER.CLIP_IMAGE_PATH,
            "box_mode_on": cfg.MODEL.MASK_FORMER.BOX_MODE_ON,
            "box_xyxy": cfg.MODEL.MASK_FORMER.BOX_XYXY,

            "tune_captioning_head": cfg.MODEL.MASK_FORMER.TUNE_CAPTIONING_HEAD,
            "captioning_thresh": cfg.MODEL.MASK_FORMER.CAPTIONING_THRESH,
            "captioning_max_out_len": cfg.MODEL.MASK_FORMER.CAPTIONING_MAX_OUT_LEN, 
            "captioning_rep_penalty": cfg.MODEL.MASK_FORMER.CAPTIONING_REPETITION_PENALTY,
            "captioning_feat_aggregation": cfg.MODEL.MASK_FORMER.MULTI_FRAME_CAPTIONING,
            "caption_num_t": cfg.MODEL.CAPTIONING_HEAD.AGGREGATION_NUM_T,
            "test_agg_middle_clip": cfg.MODEL.MASK_FORMER.TEST_AGG_MIDDLE_CLIP,
            # Experimental
            "debugging_mode_on": cfg.MODEL.MASK_FORMER.DEBUGGING_MODE_ON,
            "middle_frame_captioning": cfg.MODEL.MASK_FORMER.MIDDLE_FRAME_CAPTIONING,
            "num_captions_per_video": cfg.MODEL.MASK_FORMER.NUM_CAPTIONS_PER_VIDEO,
            "video_level_training": cfg.MODEL.MASK_FORMER.VIDEO_LEVEL_TRAINING,
            "video_matcher": matcher if cfg.MODEL.MASK_FORMER.VIDEO_LEVEL_TRAINING else None,
            "filter_person_only": cfg.MODEL.MASK_FORMER.FILTER_PERSON_ONLY,
            

            #dvoc inference
            "dvoc_inference": cfg.MODEL.MASK_FORMER.DVOC_INFERENCE,
            "dvoc_threshold": cfg.MODEL.MASK_FORMER.DVOC_INFERENCE_THRESHOLD,
            "dvoc_class_agnostic": cfg.MODEL.MASK_FORMER.DVOC_CLASS_AGNOSTIC,
            "class_agnostic_inference": cfg.MODEL.MASK_FORMER.CLASS_AGNOSTIC_INFERENCE,
            "dummy_class_prediction": cfg.MODEL.MASK_FORMER.DUMMY_CLASS_PREDICTION,
            "lvvis_zs_vidstg_class_filtering": cfg.MODEL.MASK_FORMER.LVVIS_ZS_VIDSTG_CLASS_FILTERING,
            "hysteresis_filter_enabled": cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.ENABLED,
            "hysteresis_t_low": cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.T_LOW,
            "hysteresis_t_high": cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.T_HIGH,
            "hysteresis_min_duration": cfg.MODEL.MASK_FORMER.HYSTERESIS_FILTER.MIN_DURATION,
            "per_frame_score_threshold": cfg.MODEL.MASK_FORMER.PER_FRAME_SCORE_THRESHOLD,


            # tracking
            "hungarian_track_matching_on": cfg.MODEL.MASK_FORMER.HUNGARIAN_TRACK_MATCHING,
            "hungarian_track_matcher": track_matcher,

            "topk_matching": cfg.MODEL.MASK_FORMER.TOPK_MATCHING,
            "topk_matching_num_k": cfg.MODEL.MASK_FORMER.TOPK_MATCHING_NUM_K,
            "topk_matching_num_t": cfg.MODEL.MASK_FORMER.TOPK_MATCHING_NUM_T,

            "greedy_matching": cfg.MODEL.MASK_FORMER.GREEDY_MATCHING,

            #captioning inference parameters
            "use_nucleus_sampling": cfg.MODEL.CAPTIONING_HEAD.USE_NUCLEUS_SAMPLING,
            "num_beams": cfg.MODEL.CAPTIONING_HEAD.NUM_BEAMS,
            "top_p": cfg.MODEL.CAPTIONING_HEAD.TOP_P,
            "temperature": cfg.MODEL.CAPTIONING_HEAD.TEMPERATURE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        num_frames = len(batched_inputs[0]["image"])
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        if self.debugging_mode_on :
            #### Check Image & gt
            # image = torch.stack([batched_inputs[s]['image'] for s in len(batched_inputs)]).to(rank)
            index_vis=0
            print("Video ID:", batched_inputs[index_vis]['video_id'])
            print("keys in batched_inputs:", batched_inputs[index_vis].keys())
            print("Original image shape", batched_inputs[index_vis]['width'], batched_inputs[index_vis]['height'])
            print("True image shape:", batched_inputs[index_vis]['image'][0].shape)
            print("image shape:", images[index_vis].shape)

            if 'instances' in batched_inputs[index_vis] and len(batched_inputs[index_vis]['instances']) > 0:
                
                if self.box_mode_on == False:
                    gt_masks = batched_inputs[index_vis]['instances'][0].gt_masks
                    gt_masks = gt_masks.tensor.cpu()
                    print("gt_masks shape:", gt_masks.shape)

                    # clip_index = 0
                    # gt_masks = gt_masks[clip_index].unsqueeze(0)
                    print("nb instances:", len(gt_masks))

                    # CHECK IMAGE INPUT ??
                    from captionformer.utils.visualization import Segmentation
                    filename='./TEST_MASK2FORMER_INPUT.png'
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=True)
                    # mask
                    visualization(images[index_vis].unsqueeze(0).cpu(), gt_masks.unsqueeze(0).cpu(), filename)

                
                else : 
                    gt_boxes = batched_inputs[index_vis]['instances'][0].gt_boxes
                    gt_boxes = gt_boxes.tensor.cpu()
                    # img_id = batched_inputs[index_vis]['image_id']
                    # print("image_id:", img_id)
                    print("nb instances:", len(gt_boxes))
                    
                    # CHECK IMAGE INPUT ??
                    from captionformer.utils.visualization import Segmentation
                    filename='./TEST_MASK2FORMER_INPUT.png'
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=True)
                    # # mask
                    # visualization(images[index_vis].unsqueeze(0).cpu(), gt_msk.unsqueeze(0).cpu(), filename)
                    # box
                    visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), gt_boxes.cpu(), filename)
                    # raise NotImplementedError("Visualize image[0]")
                    # bs, c, h, w = image.shape
                    # BS x C x H x W

                gt_captions = batched_inputs[index_vis]['instances'][0].gt_captions
                print("gt_captions:", gt_captions)


        features_clip = []
        if self.clip_classifier:
            clip = self.clip
            for i in range(len(batched_inputs)):
                features_clip_one_video = []
                for j in range(len(batched_inputs[i]['file_names'])):
                    file_name = 'datasets/' + batched_inputs[i]['file_names'][j].split('datasets/', 1)[1]                    
                    feature_clip = clip[file_name].to(self.device)
                    features_clip_one_video.append(feature_clip)
                features_clip_one_video = torch.cat(features_clip_one_video, dim=0)  # (T, 512)
                features_clip.append(features_clip_one_video)
            features_clip = torch.stack(features_clip, dim=1)  # (T, B, 512)

        if self.training and not self.video_level_training:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features, features_clip)
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            # bipartite matching-based loss
            losses, indices = self.criterion(outputs, targets)

            if self.debugging_mode_on :
                
                if self.box_mode_on == True :
                    from fvcore.nn import smooth_l1_loss
                    from utils.box_ops import generalized_box_iou
                    
                    print("indices:", indices)
                    def _get_src_permutation_idx(indices):
                        # permute predictions following indices
                        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
                        src_idx = torch.cat([src for (src, _) in indices])
                        return batch_idx, src_idx

                    src_idx = _get_src_permutation_idx(indices)

                    src_boxes_xywh = outputs["pred_boxes"].float()
                    


                    ###### Attn mask part
                    bs, num_queries = src_boxes_xywh.shape[:2]
                
                    from utils.box_ops import attn_masking_from_bbox
                    print("features keys : ", features.keys())
                    for k,v in features.items():
                        print("k : ", k)
                        print("x.shape : ", v.shape)
                        tgt_shape = v.shape[-2:]
                    attn_masks = attn_masking_from_bbox(src_boxes_xywh, tgt_shape , 8)
                    print("attn_masks shape : ", attn_masks.shape)
                    # bs * 8, num_queries, t*targetsize*targetsize
                    attn_masks = attn_masks.view(bs, 8, num_queries, -1)
                    print("attn_masks shape 2 : ", attn_masks.shape)
                    attn_masks = attn_masks.view(bs, 8, num_queries, -1,  tgt_shape[0], tgt_shape[1])
                    attn_masks = attn_masks.permute(1, 3, 0, 2, 4, 5)[0,0] # head 1, frame 0
                    print("attn_masks shape 3 : ", attn_masks.shape)
                    
                    print("src_idx: ",src_idx)
                    attn_masks = attn_masks[src_idx]
                    print("attn_masks shape 4 : ", attn_masks.shape)
                    # Resize to image shape
                    import numpy as np
                    import cv2

                    attn_mask_resized = F.interpolate(attn_masks.unsqueeze(0).to(torch.float), size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode='nearest')
                    attn_mask_resized = (attn_mask_resized.cpu().numpy() * 255).astype(np.uint8)
                    print("attn_mask_resized shape : ", attn_mask_resized.shape)

                    index_vis_attn_msk = 0

                    attn_mask_resized = attn_mask_resized[0,index_vis_attn_msk]
                    print("selected attn_mask_resized shape : ", attn_mask_resized.shape)
                    img_size =  images.tensor.shape[-2:]
                    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
                    img[:, :, :] = attn_mask_resized[:,:,np.newaxis]  # Set the attention mask to all channels

                    bbox = src_boxes_xywh[src_idx]
                    print("bbox_shape : ", bbox.shape)
                    print("selected bbox wh : ", bbox[index_vis_attn_msk,0])
                    # Draw the bounding box in red
                    x1, y1, w, h = bbox[index_vis_attn_msk,0]
                    x2, y2 = x1 + w, y1 + h
                    x1, y1, x2, y2 = int(x1 * img_size[1]), int(y1 * img_size[0]), int(x2 * img_size[1]), int(y2 * img_size[0])
                    print("x1, y1, x2, y2 : ", x1, y1, x2, y2)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Save the image
                    output_path = 'test_attn_mask.jpg'
                    cv2.imwrite(output_path, img)
                    ##########

                    
                    src_boxes_xywh = src_boxes_xywh[src_idx]

                    # Modified to handle video
                    target_boxes_xywh = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)]).to(src_boxes_xywh)

                    # No need to upsample predictions as we are using normalized coordinates :)
                    # NT x 4
                    src_boxes_xywh = src_boxes_xywh.flatten(0, 1)
                    target_boxes_xywh = target_boxes_xywh.flatten(0, 1)

                    # Convert from xywh to xyxy
                    src_boxes_xyxy = torch.cat([src_boxes_xywh[:, :2] , torch.max(src_boxes_xywh[:, :2], src_boxes_xywh[:, :2] + src_boxes_xywh[:, 2:])], dim=1)
                    target_boxes_xyxy = torch.cat([target_boxes_xywh[:, :2] , torch.max(target_boxes_xywh[:, :2], target_boxes_xywh[:, :2] + target_boxes_xywh[:, 2:])], dim=1)
                    # target_boxes_xywh = torch.cat([target_boxes_xyxy[:, :2] , target_boxes_xyxy[:, 2:] - target_boxes_xyxy[:, :2]], dim=1)
                    
                    # print("src_boxes", src_boxes_xywh.shape)
                    # print("target_boxes", target_boxes_xywh.shape)
                    # print("src_boxes_xyxy", src_boxes_xyxy.shape)
                    # print("target_boxes_xyxy", target_boxes_xyxy.shape)
                    

                    src_logits = outputs["pred_logits"].float()
                    
                    B, Q = src_logits.shape[0], src_logits.shape[1]
                    src_object_logits = outputs["pred_object_logits"].float()

                    idx = _get_src_permutation_idx(indices)
                    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                    target_classes = torch.full(
                        src_logits.shape[:2], self.sem_seg_head.num_classes, dtype=torch.int64, device=src_logits.device
                    )
                    target_classes[idx] = target_classes_o

                    target_object_classes = (target_classes == self.sem_seg_head.num_classes).long()
                    target_classes_binary = F.one_hot(target_classes, num_classes=self.sem_seg_head.num_classes + 1)
                    target_classes_binary = target_classes_binary[:, :, :self.sem_seg_head.num_classes].float()
                    
                    empty_object_weight = torch.ones(2, device = target_object_classes.device)
                    empty_object_weight[-1] = 0.4


                    fake_losses = {
                        "loss_giou_box": 1 - generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy).mean() if target_boxes_xyxy.numel() > 0 else torch.tensor(0.0, device=target_boxes_xywh.device),
                        "loss_l1_box": smooth_l1_loss(src_boxes_xywh, target_boxes_xywh, beta=0.0, reduction="mean"),
                        "loss_object_ce" : F.cross_entropy(src_object_logits.transpose(1, 2), target_object_classes, empty_object_weight),
                    }

                    print("losses : ", fake_losses)
                        
                    h_pad, w_pad = images.tensor.shape[-2:]


                    # Denormalize boxes
                    print("src boxes xywh :", src_boxes_xywh)
                    print("src boxes xyxy :", src_boxes_xyxy)
                    src_boxes_xywh[:,[0,2]] *= w_pad
                    src_boxes_xywh[:,[1,3]] *= h_pad
                    target_boxes_xywh[:,[0,2]] *= w_pad
                    target_boxes_xywh[:,[1,3]] *= h_pad
                    print("after scaling, image size:", w_pad, h_pad)
                    print("src boxes xywh :", src_boxes_xywh)
                    print("src boxes xyxy :", src_boxes_xyxy)

                    # CHECK IMAGE INPUT ??
                    from captionformer.utils.visualization import Segmentation
                    filename='./TEST_MASK2FORMER_GT.png'
                    
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # # mask
                    # visualization(images[index_vis].unsqueeze(0).cpu(), gt_msk.unsqueeze(0).cpu(), filename)
                    print("target boxes shape:", target_boxes_xywh.shape)
                    # box
                    visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), target_boxes_xywh.cpu(), filename)
                    
                    filename='./TEST_MASK2FORMER_PRED.png'
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # # # mask
                    # # visualization(images[index_vis].unsqueeze(0).cpu(), gt_msk.unsqueeze(0).cpu(), filename)
                    print("src boxes shape:", src_boxes_xywh.shape)
                    # box
                    visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), src_boxes_xywh.cpu(), filename)
                    print("Saved prediction at :", filename)
                    print("\n\n")
                    from time import sleep
                    # sleep(1)

                    # If n_objects>0 raise error
                    if target_boxes_xywh.numel() > 0:
                        raise NotImplementedError("Losses")
                    
                self.i_debug += 1
                if self.i_debug == 1:
                    raise NotImplementedError("Stop")



            # Captioning loss
            if self.mask_captioning:
                pred_queries = outputs["pred_queries"]
                pred_masks_or_boxes = outputs["pred_masks"] if not self.box_mode_on else outputs["pred_boxes"]

                fnames = [x["file_names"] for x in batched_inputs]
                feature_ids = []
                for fname  in fnames:
                    vid_ids = [os.path.basename(os.path.dirname(f).split('/imgs')[0]) for f in fname]

                    assert all(x == vid_ids[0] for x in vid_ids)
                    vid_ids = vid_ids[0]

                    # remove extension as well
                    indexes = [int(os.path.splitext(os.path.basename(f))[0]) for f in fname]
                    feature_ids.append({"video_id": vid_ids, "frame_index": indexes})
                
                caption_loss = self.captioning_head({"image": images.tensor,  "target" : targets, "feature_id" : feature_ids}, pred_queries, pred_masks_or_boxes, None, indices)

                losses["loss_caption"] = caption_loss


            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] = losses[k] * self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            
            return losses
        else:
            # Initialize timing variables
            video_start_time = detection_start_time = detection_end_time = None
            tracking_start_time = tracking_end_time = None
            captioning_start_time = captioning_end_time = video_end_time = None
            
            # Start timing for this video
            if self.enable_timing:
                self.timing_video_count += 1
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                video_start_time = time.perf_counter()
                detection_start_time = time.perf_counter()
            
            # c = 5  # clip length
            c = self.inference_clip_len # clip length
            mask_cls = []
            mask_obj = []
            mask_pred = []
            box_pred = []
            mask_embed = []
            for i in range(math.ceil(num_frames / c)):
                image_c = images.tensor[i * c: (i + 1) * c]
                if self.clip_classifier:
                    features_clip_c = features_clip[i * c: (i + 1) * c]
                else :
                    features_clip_c=None
                with torch.no_grad():
                    features = self.backbone(image_c)
                    outputs = self.sem_seg_head(features, features_clip_c)
                
                mask_cls.append(outputs["pred_logits"][0])
                mask_obj.append(outputs["pred_object_logits"][0])
                mask_pred.append(outputs["pred_masks"][0]) if not self.box_mode_on else box_pred.append(outputs["pred_boxes"][0])
                mask_embed.append(outputs["pred_embds"][0])
                
            del outputs, features, image_c, features_clip_c

            # End detection timing, start tracking timing
            if self.enable_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                detection_end_time = time.perf_counter()
                tracking_start_time = time.perf_counter()

            out_cls = []
            out_obj = []
            out_pred = []
            out_embed = []
            out_cls.append(mask_cls[0])
            out_obj.append(mask_obj[0])

            out_pred.append(mask_pred[0] if not self.box_mode_on else box_pred[0])
            out_embed.append(mask_embed[0])
            
            # Match embeddings (tracking) and store scores to select captioned clip
            scores = torch.zeros((out_cls[0].shape[0],len(mask_cls)), dtype=torch.float, device=out_cls[0].device)
            scores_cls = torch.zeros((out_cls[0].shape[0],len(mask_cls)), dtype=torch.float, device=out_cls[0].device)
            object_scores = torch.zeros((out_cls[0].shape[0],len(mask_cls)), dtype=torch.float, device=out_cls[0].device)
            # print("scores shape:", scores.shape)
            score_0, scores_cls_0, object_scores_0 = self.get_pred_score(out_cls[0],out_obj[0])
            # print("score_0 shape:", score_0.shape)
            scores[:,0] = score_0.max(dim=1).values
            scores_cls[:,0] = scores_cls_0.max(dim=1).values
            object_scores[:,0] = object_scores_0.max(dim=1).values
            if self.topk_matching:
                # to device
                embed_memory=deque(maxlen=self.topk_matching_num_t)
                embed_memory.append(out_embed[0])
            elif self.greedy_matching :
                # Asso matrix : similarity matrix with sigmoid
                asso_matrix=self.get_association_matrix(mask_embed)
                # values between 0 and 1 strictly so force matching between all queries
                grid_ids=self.greedy_extract_trajectories(asso_matrix,num_frames=len(mask_embed))
                del asso_matrix
                
                out_cls = [out_cls[0][grid_ids[0], :]]
                out_obj = [out_obj[0][grid_ids[0], :]]
                out_pred = [out_pred[0][grid_ids[0], :, :, :] if not self.box_mode_on else box_pred[0][grid_ids[0], :, :]]
                out_embed = [out_embed[0][grid_ids[0], :]]

            for i in range(1, len(mask_cls)):
                if self.topk_matching:
                    indices = self.match_from_vote_embds(embed_memory,mask_embed[i])
                    embed_memory.append(mask_embed[i][indices, :])
                elif self.greedy_matching:
                    indices = grid_ids[i]


                elif self.hungarian_track_matching_on:
                    assert self.box_mode_on == True
                    src = {"pred_logits": out_cls[-1], 
                           "pred_object_logits": out_obj[-1], 
                           "pred_boxes": out_pred[-1], 
                           "pred_embds": out_embed[-1]}
                    tgt = {"pred_logits": mask_cls[i],
                            "pred_object_logits": mask_obj[i],
                            "pred_boxes": box_pred[i],
                            "pred_embds": mask_embed[i]}
                    indices = self.hungarian_track_matcher(src, tgt)

                else :
                    indices = self.match_from_embds(out_embed[-1].detach(), mask_embed[i].detach())
                
                out_cls.append(mask_cls[i][indices, :])
                out_obj.append(mask_obj[i][indices, :])
                out_pred.append(mask_pred[i][indices, :, :, :] if not self.box_mode_on else box_pred[i][indices, :, :])
                out_embed.append(mask_embed[i][indices, :])
                # memory
                # alpha = 0.7
                # tmp_pred_embds = alpha * mask_embed[i][indices, :] + (1 - alpha) * out_embed[-1]
                # out_embed.append(tmp_pred_embds)
                score_i, scores_cls_i, object_scores_i = self.get_pred_score(out_cls[-1],out_obj[-1])
                scores[:,i] = score_i.max(dim=1).values
                scores_cls[:,i] = scores_cls_i.max(dim=1).values
                object_scores[:,i] = object_scores_i.max(dim=1).values
            # nquery, nclip
            # Get the max index for each query 
            max_score, max_score_index = scores.max(dim=1)

            if self.middle_frame_captioning:
                num_clips = num_frames // c
                # use middle frame
                mid_frame_index = num_clips // 2
                max_score_index = torch.ones_like(max_score_index) * mid_frame_index

            mask_cls_result = sum(out_cls) / len(out_cls)  # (100,1197)
            mask_obj_result = sum(out_obj) / len(out_obj)  # (100,2)

            # End tracking timing, start captioning timing  
            if self.enable_timing:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                tracking_end_time = time.perf_counter()
                captioning_start_time = time.perf_counter()

            if self.training :
                assert self.mask_captioning, "Video-level training only implemented for captioning."
                if self.mask_captioning :
                    # assert self.box_mode_on == True
                    
                    # Create empty prediction and iterate over the clip to aggregate pres_masks/pred_boxes (concat on axis 1) and class score (select mean to choose class)
                    tgt_prep_full = []
                    if self.test_agg_middle_clip :
                        num_clips = num_frames // c
                        # use middle frame
                        mid_frame_index = num_clips // 2
                        clip_indices = torch.tensor([mid_frame_index])
                    else :
                        clip_indices = torch.linspace(0, len(out_embed) - 1, steps=self.caption_num_t).long()

                    fnames = [x["file_names"] for x in batched_inputs]
                    selected_img_clip=[]
                    selected_img_sizes=[]
                    selected_pred = []
                    selected_embed = []
                    selected_scores = []
                    selected_tgt = []
                    selected_feature_ids = []

                    matching_ignore_clips=[]
                    
                    for i in range(math.ceil(num_frames / c)):
                        # prepare tgt
                        img_clip = images.tensor[i * c: (i + 1) * c]
                        img_size=images.image_sizes[i * c: (i + 1) * c]
                        if i in clip_indices:
                            selected_img_clip.append(img_clip)
                            selected_img_sizes.append(img_size)
                            selected_embed.append(out_embed[i].unsqueeze(0))
                            selected_scores.append(scores[:,i])
                            selected_pred.append(out_pred[i].unsqueeze(0))
                            for fname in fnames: 
                                vid_ids = [os.path.basename(os.path.dirname(f).split('/imgs')[0]) for f in fname]
                                assert all(x == vid_ids[0] for x in vid_ids)
                                vid_ids = vid_ids[0]
                                feat_indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in fname[i * c: (i + 1) * c]]
                                selected_feature_ids.append([{"video_id": vid_ids, "frame_index": feat_indices}])

                        img_clip = ImageList(img_clip, image_sizes=img_size)                        
                        batched_inputs_clip = [{'height':x['height'],
                                                'width':x['width'],
                                                'length':x['length'],
                                                'video_id':x['video_id'],
                                                'image':x['image'][i * c: (i + 1) * c],
                                                'instances':x['instances'][i * c: (i + 1) * c] if 'instances' in x else None,
                                                'file_names':x['file_names'][i * c: (i + 1) * c] if 'file_names' in x else None,
                                                } for x in batched_inputs]                        

                        num_instances = len(batched_inputs_clip[0]['instances'][0]) if 'instances' in batched_inputs_clip[0] else 0

                        tgt_prep=self.prepare_targets(batched_inputs_clip, img_clip, video_level = True)
                        num_instances_after = len(tgt_prep[0]['labels']) if 'labels' in tgt_prep[0] else 0

                        if i in clip_indices:
                            selected_tgt.append(tgt_prep)
                        if num_instances_after != num_instances:
                            matching_ignore_clips.append(i)
                            continue
                        if len(tgt_prep_full) == 0:
                            tgt_prep_full = tgt_prep
                        else :
                            for id_tgt,x,y in zip(range(len(tgt_prep_full)), tgt_prep_full, tgt_prep):
                                if self.box_mode_on == False:
                                    tgt_prep_full[id_tgt]['masks'] = torch.cat((x['masks'], y['masks']), dim=1)
                                else :
                                    tgt_prep_full[id_tgt]['boxes'] = torch.cat((x['boxes'], y['boxes']), dim=1)
                                tgt_prep_full[id_tgt]['ids'] = torch.cat((x['ids'], y['ids']), dim=1)
                                
                    del batched_inputs, batched_inputs_clip, tgt_prep, img_clip
                   
                    # prepare predictions
                    out_pred = [out_pred[i] for i in range(len(out_pred)) if i not in matching_ignore_clips]
                    out_pred_full = torch.cat(out_pred, dim=1).unsqueeze(0)
                   
                    if self.box_mode_on == False:
                        pred_prep_full = {
                            "pred_logits": mask_cls_result.unsqueeze(0),  # (B, Q, C)
                            "pred_object_logits": mask_obj_result.unsqueeze(0),  # (B, Q, C)
                            "pred_masks": out_pred_full,  # (T, B, Q, H, W)
                        }
                    else :
                        pred_prep_full = {
                            "pred_logits": mask_cls_result.unsqueeze(0),  # (B, Q, C)
                            "pred_object_logits": mask_obj_result.unsqueeze(0),  # (B, Q, C)
                            "pred_boxes": out_pred_full,  # (T, B, Q, 4)
                        }
                        
                    # Get matching
                    indices = self.video_matcher(pred_prep_full, tgt_prep_full)
                    del pred_prep_full, tgt_prep_full

                    selected_samples = [{"image": x, "target": y, "feature_id": z} for x, y, z in zip(selected_img_clip, selected_tgt, selected_feature_ids)]
                    
                    caption_loss = self.captioning_head(selected_samples, selected_embed, selected_pred, selected_scores, indices)

                    losses = {"loss_caption": caption_loss}
                else :
                    raise NotImplementedError("video level training but no video level losses")
                

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] = losses[k] * self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
               
                return losses
            else :
                # Captioning head
                if self.mask_captioning:
                    # get selected clips
                    if self.num_captions_per_video > 1:
                        out_captions = [[''] * num_frames for _ in range(max_score_index.shape[0])]
                        
                        for part in range(self.num_captions_per_video):
                            idx_start = num_frames * part // self.num_captions_per_video
                            idx_end = num_frames * (part + 1) // self.num_captions_per_video
                            idx_clip_start = idx_start // c
                            idx_clip_end = idx_end // c
                            scores_part = torch.zeros_like(scores, dtype=torch.float, device=scores.device)
                            scores_part[:, idx_clip_start:idx_clip_end] = scores[:, idx_clip_start:idx_clip_end]
                            max_score_part, max_score_index_part = scores_part.max(dim=1)
                            
                            out_cap=None
                            for qid in range(max_score_index_part.shape[0]):
                                max_score_cur = max_score_part[qid]
                                if max_score_cur < self.captioning_thresh:
                                    # out_captions_part.append('')
                                    # out_captions[qid][idx_start:idx_end] = ''
                                    continue
                            
                                clip_selected = max_score_index_part[qid]
                                pred_q = out_embed[clip_selected][qid].unsqueeze(0).unsqueeze(0)
                                pred_m = out_pred[clip_selected][qid].unsqueeze(0).unsqueeze(0)

                                fnames = [x["file_names"] for x in batched_inputs]
                                feature_ids = []
                                for fname  in fnames:
                                    vid_ids = [os.path.basename(os.path.dirname(f).split('/imgs')[0]) for f in fname[clip_selected * c: (clip_selected + 1) * c]]
                                    assert all(x == vid_ids[0] for x in vid_ids)
                                    vid_ids = vid_ids[0]

                                    # remove extension as well
                                    indexes = [int(os.path.splitext(os.path.basename(f))[0]) for f in fname[clip_selected * c: (clip_selected + 1) * c]]
                                    feature_ids.append({"video_id": vid_ids, "frame_index": indexes})

                                sample = {
                                    "image": images.tensor[clip_selected * c: (clip_selected + 1) * c],
                                    "text_input" : ["a photo of"],
                                    "feature_id" : feature_ids
                                }
                                out_cap = self.captioning_head.generate(sample, pred_q, pred_m,
                                    max_length=self.captioning_max_out_len,
                                    repetition_penalty=self.captioning_rep_penalty,
                                    use_nucleus_sampling=self.captioning_use_nucleus_sampling,
                                    num_beams=self.captioning_num_beams,
                                    top_p=self.captioning_top_p,
                                    temperature=self.captioning_temperature
                                )
                                
                                for idx in range(idx_start, idx_end):
                                    out_captions[qid][idx] = out_cap[-1][-1]

                                # in case of more cap per vid than number of frames, need to break for the last frames
                                if idx_end >= num_frames:
                                    break
                        
                    else :    
                        out_captions = []
                        out_cap=None
                        for qid in range(max_score_index.shape[0]):
                            max_score_cur = max_score[qid]
                            if max_score_cur < self.captioning_thresh:
                                out_captions.append('')
                                continue
                            
                            if self.captioning_feat_aggregation:

                                if self.test_agg_middle_clip :
                                    num_clips = num_frames // c
                                    # use middle frame
                                    mid_frame_index = num_clips // 2
                                    clip_indices = torch.tensor([mid_frame_index])
                                else :
                                    # sample T clips uniformly in the video
                                    clip_indices = torch.linspace(0, len(out_embed) - 1, steps=self.caption_num_t).long()

                                pred_qs = []
                                pred_ms = []
                                pred_ss = []
                                cap_input = []
                                for clip_selected in clip_indices:
                                    clip_selected = clip_selected.item()
                                    pred_qs.append(out_embed[clip_selected][qid].unsqueeze(0).unsqueeze(0))
                                    pred_ms.append(out_pred[clip_selected][qid].unsqueeze(0).unsqueeze(0))

                                    pred_ss.append(scores[qid][clip_selected])
                                    cap_input.append({
                                        "image": images.tensor[clip_selected * c: (clip_selected + 1) * c],
                                        "text_input" : ["a photo of"]
                                    })

                                out_cap = self.captioning_head.generate_video(
                                    cap_input, 
                                    pred_qs, 
                                    pred_ms,
                                    pred_ss,
                                    max_length=self.captioning_max_out_len,
                                    repetition_penalty=self.captioning_rep_penalty
                                )

                                out_captions.append(out_cap[-1])

                            else :
                                clip_selected = max_score_index[qid]
                                pred_q = out_embed[clip_selected][qid].unsqueeze(0).unsqueeze(0)
                                pred_m = out_pred[clip_selected][qid].unsqueeze(0).unsqueeze(0)

                                fnames = [x["file_names"] for x in batched_inputs]
                                feature_ids = []
                                for fname  in fnames:
                                    vid_ids = [os.path.basename(os.path.dirname(f).split('/imgs')[0]) for f in fname[clip_selected * c: (clip_selected + 1) * c]]
                                    assert all(x == vid_ids[0] for x in vid_ids)
                                    vid_ids = vid_ids[0]

                                    # remove extension as well
                                    indexes = [int(os.path.splitext(os.path.basename(f))[0]) for f in fname[clip_selected * c: (clip_selected + 1) * c]]
                                    feature_ids.append({"video_id": vid_ids, "frame_index": indexes})
                                
                                sample = {
                                    "image": images.tensor[clip_selected * c: (clip_selected + 1) * c],
                                    "text_input" : ["a photo of"],
                                    "feature_id" : feature_ids
                                    }
                                out_cap = self.captioning_head.generate(sample, pred_q, pred_m, 
                                    max_length=self.captioning_max_out_len,
                                    
                                    repetition_penalty=self.captioning_rep_penalty,
                                    
                                    use_nucleus_sampling=self.captioning_use_nucleus_sampling,
                                    num_beams=self.captioning_num_beams,
                                    top_p=self.captioning_top_p,
                                    temperature=self.captioning_temperature
                                    )
                                out_captions.append(out_cap[-1])
                                
                        del max_score_index, out_cap
                else :
                    out_captions = None

                # End captioning timing, start post-processing timing
                if self.enable_timing:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    captioning_end_time = time.perf_counter()
                    post_processing_start_time = time.perf_counter()
                
                input_per_image = batched_inputs[0]
                image_size = images.image_sizes[0]  # image size without padding after data augmentation

                height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
                width = input_per_image.get("width", image_size[1])

                if not self.box_mode_on :
                    mask_pred_result = torch.cat(out_pred, dim=1)  # (100,T,h,w)                
                    
                    # Offline

                    mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                        mask_pred_result,
                        size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    ).to(self.device)
                    
                    del mask_cls, mask_pred, box_pred, mask_embed, out_pred, out_embed

                    if self.dvoc_inference:
                        results = retry_if_cuda_oom(self.inference_video_dvoc_mask)(out_cls, mask_cls_result, out_obj, mask_obj_result, mask_pred_result, out_captions, image_size, height, width)
                    else :
                        results = retry_if_cuda_oom(self.inference_video_instance)(mask_cls_result, mask_obj_result, mask_pred_result, out_captions, image_size, height, width)
                    
                    # End post-processing timing and save statistics
                    if self.enable_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        post_processing_end_time = time.perf_counter()
                        video_end_time = time.perf_counter()
                        
                        # Calculate times in milliseconds
                        detection_time = (detection_end_time - detection_start_time) * 1000
                        tracking_time = (tracking_end_time - tracking_start_time) * 1000
                        captioning_time = (captioning_end_time - captioning_start_time) * 1000
                        post_processing_time = (post_processing_end_time - post_processing_start_time) * 1000
                        total_time = (video_end_time - video_start_time) * 1000
                        
                        self.timing_stats['detection_times'].append(detection_time)
                        self.timing_stats['tracking_times'].append(tracking_time)
                        self.timing_stats['captioning_times'].append(captioning_time)
                        self.timing_stats['post_processing_times'].append(post_processing_time)
                        self.timing_stats['total_times'].append(total_time)
                        self.timing_stats['num_frames_per_video'].append(num_frames)
                        
                        # Store per-video stats
                        video_stats = {
                            'video_id': self.timing_video_count,
                            'num_frames': num_frames,
                            'detection_ms': detection_time,
                            'tracking_ms': tracking_time,
                            'captioning_ms': captioning_time,
                            'post_processing_ms': post_processing_time,
                            'total_ms': total_time,
                            'detection_fps': num_frames / (detection_time / 1000),
                            'tracking_fps': num_frames / (tracking_time / 1000),
                            'captioning_fps': num_frames / (captioning_time / 1000),
                            'post_processing_fps': num_frames / (post_processing_time / 1000),
                            'total_fps': num_frames / (total_time / 1000),
                        }
                        self.timing_stats['per_video_stats'].append(video_stats)
                        
                        # Add timing info to results
                        results['timing'] = {
                            'detection_ms': detection_time,
                            'tracking_ms': tracking_time,
                            'captioning_ms': captioning_time,
                            'post_processing_ms': post_processing_time,
                            'total_ms': total_time,
                            'num_frames': num_frames
                        }
                    
                    if self.debugging_mode_on:
                        index_vis = 0
                        vid_id = input_per_image['video_id']
                        print("Video ID:", vid_id)

                        pred_masks = results["pred_masks"]
                        pred_labels = results["pred_labels"]
                        pred_scores = torch.tensor(results["pred_scores"])

                        # print("pred_boxes:", pred_boxes)
                        print("len pred_masks:", len(pred_masks))
                        # print("shape pred_boxes:", pred_boxes.shape)
                        print("pred_labels:", pred_labels)
                        print("pred_scores:", pred_scores)
                        print("len :", len(pred_labels))

                        print("captions :", out_captions)

                        # Filter predictions
                        thresh = 0.6
                        # keep = (pred_scores > thresh)
                        # keep_indices = torch.nonzero(keep).squeeze(1)
                        pred_masks = [x.unsqueeze(0) for x,y in zip(pred_masks, pred_scores) if y > thresh]
                        # pred_labels = [x for x,y in zip(pred_labels, pred_scores) if y > thresh]
                        # pred_scores = [x for x in pred_scores if x > thresh]

                        img = images[index_vis].unsqueeze(0)
                        print("img plt shape:", img.shape)
                        img = F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)
                        # img = img.to(torch.int)
                        print("img shape:", img.shape)
                        if pred_masks:
                            # Cat on a new dim 0
                            pred_masks = torch.cat(pred_masks, dim=0)
                        else :
                            pred_masks = torch.empty((height, width), dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
                        # pred_boxes = pred_boxes[keep]
                        # pred_labels = pred_labels[keep]
                        # pred_scores = pred_scores[keep]
                        print("pred_masks:", pred_masks)
                        print("pred_masks:", pred_masks.shape)
                        print("number of non 0 pred_masks:", (pred_masks[index_vis].sum(dim=(1,2)) > 0).sum())
                        print("pred_labels:", pred_labels)
                        print("pred_scores:", pred_scores)

                        
                        # # Reconvert bbox
                        # pred_boxes = pred_boxes.to(torch.float)  # Convert to float before scaling
                        # ratio_h = input_height / height
                        # ratio_w = input_width / width
                        # pred_boxes[:,:,[0,2]] *= ratio_w
                        # pred_boxes[:,:,[1,3]] *= ratio_h
                        # print("good ratio pred boxes ", pred_boxes)
                        # # print("shape pred_boxes:", pred_boxes.shape)
                        # print("pred_masks shape:", pred_masks.shape)

                        # CHECK IMAGE INPUT ??
                        from captionformer.utils.visualization import Segmentation
                        filename='./TEST_MASK2FORMER_PRED.png'
                        visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                        # mask
                        visualization(img.cpu(), pred_masks[index_vis].unsqueeze(0).cpu(), filename)
                        # # box
                        # visualization.plot_bbox(img.cpu(), pred_boxes[:,index_vis].cpu(), filename)

                        self.i_debug += 1
                        if self.i_debug >= 2:
                            raise Exception("Visualization : debugging mode")
                
                    return results
                else :
                    box_pred_result = torch.cat(out_pred, dim=1) # (100,T,4)
                    # Offline
                    
                    input_height, input_width =  image_size
                    pad_h, pad_w = images.tensor.shape[-2:]

                    # Convert from relative coordinates [0,1] with padding to relative [0,1] without padding
                    ratio_h = pad_h / input_height
                    ratio_w = pad_w / input_width
                    box_pred_result[:, :, [0, 2]] *= ratio_w
                    box_pred_result[:, :, [1, 3]] *= ratio_h
                    # Clamp values to (0, 1)
                    box_pred_result[:, :, [0, 2]] = box_pred_result[:, :, [0, 2]].clamp(0, 1)
                    box_pred_result[:, :, [1, 3]] = box_pred_result[:, :, [1, 3]].clamp(0, 1)

                    del mask_cls, mask_pred, box_pred, mask_embed, out_pred, out_embed

                    if self.dvoc_inference:
                        results = retry_if_cuda_oom(self.inference_video_dvoc)(out_cls, mask_cls_result, out_obj, mask_obj_result, box_pred_result, out_captions, image_size, height, width)
                    else :
                        results = retry_if_cuda_oom(self.inference_video_instance)(mask_cls_result, mask_obj_result, box_pred_result, out_captions, image_size, height, width)
                    
                    # End post-processing timing and save statistics
                    if self.enable_timing:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        post_processing_end_time = time.perf_counter()
                        video_end_time = time.perf_counter()
                        
                        # Calculate times in milliseconds
                        detection_time = (detection_end_time - detection_start_time) * 1000
                        tracking_time = (tracking_end_time - tracking_start_time) * 1000
                        captioning_time = (captioning_end_time - captioning_start_time) * 1000
                        post_processing_time = (post_processing_end_time - post_processing_start_time) * 1000
                        total_time = (video_end_time - video_start_time) * 1000
                        
                        self.timing_stats['detection_times'].append(detection_time)
                        self.timing_stats['tracking_times'].append(tracking_time)
                        self.timing_stats['captioning_times'].append(captioning_time)
                        self.timing_stats['post_processing_times'].append(post_processing_time)
                        self.timing_stats['total_times'].append(total_time)
                        self.timing_stats['num_frames_per_video'].append(num_frames)
                        
                        # Store per-video stats
                        video_stats = {
                            'video_id': self.timing_video_count,
                            'num_frames': num_frames,
                            'detection_ms': detection_time,
                            'tracking_ms': tracking_time,
                            'captioning_ms': captioning_time,
                            'post_processing_ms': post_processing_time,
                            'total_ms': total_time,
                            'detection_fps': num_frames / (detection_time / 1000),
                            'tracking_fps': num_frames / (tracking_time / 1000),
                            'captioning_fps': num_frames / (captioning_time / 1000),
                            'post_processing_fps': num_frames / (post_processing_time / 1000),
                            'total_fps': num_frames / (total_time / 1000),
                        }
                        self.timing_stats['per_video_stats'].append(video_stats)
                        
                        # Add timing info to results
                        results['timing'] = {
                            'detection_ms': detection_time,
                            'tracking_ms': tracking_time,
                            'captioning_ms': captioning_time,
                            'total_ms': total_time,
                            'num_frames': num_frames
                        }
                    
                    if self.debugging_mode_on:
                        index_vis = 0
                        vid_id = input_per_image['video_id']
                        print("Video ID:", vid_id)

                        pred_boxes = results["pred_boxes"]
                        # pred_boxes = torch.cat([x[index_vis] for x in pred_boxes], dim=1)
                        pred_labels = results["pred_labels"]
                        pred_scores = torch.tensor(results["pred_scores"])
                        if self.mask_captioning:   
                            pred_caption = results["pred_captions"]
                        else :
                            pred_caption = [None] * len(pred_scores)
                        # print("pred_boxes:", pred_boxes)
                        print("index vis element of all bbox:", [x[index_vis] for x in pred_boxes])
                        print("len pred_boxes:", len(pred_boxes))
                        # print("shape pred_boxes:", pred_boxes.shape)
                        print("pred_labels:", pred_labels)
                        print("pred_scores:", pred_scores)
                        print("len :", len(pred_labels))
                        print("out_captions :", out_captions)
                        print("pred_caption:", pred_caption)
                        # Filter predictions
                        thresh = 0.3
                        keep = (pred_scores > thresh)
                        print("num kept queries :", keep.sum())
                        # keep_indices = torch.nonzero(keep).squeeze(1)
                        idx_pred_boxes = [m[index_vis] for m,y in zip(pred_boxes, pred_scores) if (y > thresh and m[index_vis] is not None) ]
                        pred_labels = [x for x,y,m in zip(pred_labels, pred_scores, pred_boxes) if (y > thresh and m[index_vis] is not None)]
                        pred_caption = [x for x,y,m in zip(pred_caption, pred_scores, pred_boxes) if (y > thresh and m[index_vis] is not None)]
                        pred_scores = [y for y,m in zip(pred_scores,pred_boxes) if (y > thresh and m[index_vis] is not None)]
                        print("num visible in frame {} : {}".format(index_vis, len(pred_boxes)))

                        print("idx_pred_boxes:", idx_pred_boxes)
                        print("len idx_pred_boxes:", len(idx_pred_boxes))
                        if len(idx_pred_boxes) > 0:
                            print("len idx_pred_boxes[0]:", len(idx_pred_boxes[0]))
                        if idx_pred_boxes:
                            # Cat on a new dim 0
                            # idx_pred_boxes = torch.tensor(idx_pred_boxes)
                            idx_pred_boxes = torch.stack([torch.tensor(x) for x in idx_pred_boxes], dim=0)
                        else :
                            idx_pred_boxes = torch.empty((0, 4))
                        
                        # pred_caption = [x for x,y in zip(pred_caption, pred_scores) if y > thresh]

                        # pred_boxes = pred_boxes[keep]
                        # pred_labels = pred_labels[keep]
                        # pred_scores = pred_scores[keep]
                        # print("pred_boxes:", idx_pred_boxes)
                        print("pred_boxes:", idx_pred_boxes.shape)
                        print("pred_labels:", pred_labels)
                        print("pred_scores:", pred_scores)
                        print("pred_caption:", pred_caption)

                        img = images[index_vis].unsqueeze(0)
                        print("img plt shape:", img.shape)
                        # img = F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)
                        # img = img.to(torch.int)
                        # Reconvert bbox
                        idx_pred_boxes = idx_pred_boxes.to(torch.float)  # Convert to float before scaling
                        ratio_h = input_height / height
                        ratio_w = input_width / width
                        idx_pred_boxes[:,[0,2]] *= ratio_w
                        idx_pred_boxes[:,[1,3]] *= ratio_h
                        print("good ratio pred boxes ", idx_pred_boxes)
                        # print("shape pred_boxes:", pred_boxes.shape)


                        # CHECK IMAGE INPUT ??
                        from captionformer.utils.visualization import Segmentation
                        filename='./TEST_MASK2FORMER_PRED.png'
                        visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                        # # mask
                        # visualization(images[index_vis].unsqueeze(0).cpu(), gt_msk.unsqueeze(0).cpu(), filename)
                        # box
                        visualization.plot_bbox(img.cpu(), idx_pred_boxes.cpu(), filename)
                        self.i_debug += 1
                        if self.i_debug == 4:
                            raise Exception("Visualization : debugging mode")
                    
                    return results

    def prepare_targets(self, targets, images, video_level=False, return_no_cls_ids=False):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])

            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            box_shape = [_num_instance, self.num_frames, 4]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            gt_boxes_per_video = torch.zeros(box_shape, dtype=torch.float32, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                if not self.box_mode_on:
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else :
                    gt_boxes_per_video[:, f_i, :] = targets_per_frame.gt_boxes.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)

            if video_level:
                # all ids are valid
                valid_idx = gt_ids_per_video.any(dim=-1)
            else :
                valid_idx = (gt_ids_per_video != -1).any(dim=-1)
            
            
            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames
            indices = torch.nonzero(valid_idx).squeeze(1).tolist()
            
            dataset_name = targets_per_video.get("dataset", "unknown") 
            instance_dict = {
                "labels": gt_classes_per_video,
                "ids": gt_ids_per_video,
                "dataset": dataset_name  # <-- New field to identify dataset origin
            }
            if self.mask_captioning:
                gt_captions_per_video = [targets_per_frame.gt_captions[i] for i in indices]  # N,
                instance_dict["captions"] = gt_captions_per_video
            if not self.box_mode_on:
                gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
                instance_dict["masks"] = gt_masks_per_video
            else:
                gt_boxes_per_video = gt_boxes_per_video[valid_idx]                  # N, num_frames, 4
                # Normalize boxes
                gt_boxes_per_video[:, :, [0, 2]] /= w_pad
                gt_boxes_per_video[:, :, [1, 3]] /= h_pad
                instance_dict["boxes"] = gt_boxes_per_video

            gt_instances.append(instance_dict)

        return gt_instances

    def inference_video_instance(self, pred_cls, pred_obj, pred_masks_or_boxes, mask_cap_results, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            if self.filter_person_only:
                scores = pred_cls[:, 772:773].sigmoid()  
            else :
                scores = pred_cls[:, :-1].sigmoid()

            object_scores = F.softmax(pred_obj, dim=-1)[:, :-1]

            if self.class_agnostic_inference:
                scores = scores
            else :
                scores = (scores * object_scores) ** 0.5

            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(50, sorted=False)
            
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            
            # Sort captions with topk indices
            if self.mask_captioning:
                out_caps = [mask_cap_results[i] for i in topk_indices]
            elif self.dummy_captioning:
                # caption should be the category name
                out_caps = [self.category_names[label] for label in labels_per_image]
                
            else :
                out_caps = []

            if not self.box_mode_on:
                pred_masks = pred_masks_or_boxes[topk_indices]

                pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
                pred_masks = F.interpolate(
                    pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
                )
                
                
                masks = pred_masks > 0.
                out_masks = [m for m in masks.cpu()]
            else :
                pred_boxes = pred_masks_or_boxes[topk_indices]
                
                # Denormalize boxes
                pred_boxes[:, :, [0, 2]] *= output_width 
                pred_boxes[:, :, [1, 3]] *= output_height
    
                if self.box_xyxy:
                    # Convert boxes to xyxy
                    from utils.box_ops import box_xywh_to_xyxy
                    pred_boxes = torch.cat([box_xywh_to_xyxy(x).unsqueeze(0) for x in pred_boxes], dim=0)

                # Round to int
                out_boxes = [m.round().to(torch.int) for m in pred_boxes.cpu()]
            
            if self.dummy_class_prediction:
                labels_per_image = torch.zeros_like(labels_per_image)
            elif self.lvvis_zs_vidstg_class_filtering:
                # Filter and map LVVIS categories to VidSTG
                # Check that labels are in valid range
                assert labels_per_image.max() < 1196, f"Label {labels_per_image.max()} exceeds LVVIS range (0-1195)"
                # Create mask of predictions to keep (only those in mapping)
                keep_mask = torch.tensor([label.item() in self.lvvis_to_vidstg_mapping for label in labels_per_image], device=labels_per_image.device)
                # Map LVVIS labels to VidSTG labels for kept predictions
                mapped_labels = torch.tensor([self.lvvis_to_vidstg_mapping[label.item()] for label in labels_per_image[keep_mask]], device=labels_per_image.device)
                # Filter all predictions
                labels_per_image = mapped_labels
                scores_per_image = scores_per_image[keep_mask.cpu()]
                if self.box_mode_on:
                    out_boxes = [box for box, keep in zip(out_boxes, keep_mask.cpu().tolist()) if keep]
                else:
                    out_masks = [mask for mask, keep in zip(out_masks, keep_mask.cpu().tolist()) if keep]
                out_caps = [cap for cap, keep in zip(out_caps, keep_mask.cpu().tolist()) if keep]
            
            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            
        else:
            out_scores = []
            out_labels = []
            out_caps = []
            out_masks = []
            out_boxes = []
        
        if not self.box_mode_on:
            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_masks": out_masks,
            }
        else :
            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_boxes": out_boxes,
            }
        if self.mask_captioning or self.dummy_captioning:
            video_output["pred_captions"] = out_caps

        video_output = self._apply_hysteresis_filter(video_output)
        video_output = self._apply_per_frame_score_threshold(video_output)
        return video_output

    def inference_video_dvoc_mask(self, per_img_pred_cls, pred_cls, per_img_pred_obj, pred_obj, pred_masks, mask_cap_results, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            assert not self.box_mode_on
            # Stack predictions into tensors
            per_img_pred_cls = torch.stack(per_img_pred_cls, dim=0)  # (nclips, nqueries, nclasses)
            per_img_pred_obj = torch.stack(per_img_pred_obj, dim=0)  # (nclips, nqueries, 2)

            # Compute object and class scores
            per_img_object_scores = F.softmax(per_img_pred_obj, dim=-1)[:, :, 0]  # (nclips, nqueries)
            video_object_scores = per_img_object_scores.mean(dim=0)  # (nqueries,)
            if self.filter_person_only:
                # Filter out all classes except person (score=0)
                per_img_pred_cls = per_img_pred_cls[:, :, 772:773]  # (nclips, nqueries, 1)
                per_img_scores, _ = per_img_pred_cls[:, :, :].sigmoid().max(dim=-1) # (nclips, nqueries)
                video_scores, labels = per_img_pred_cls[:, :, :].mean(dim=0).sigmoid().max(dim=-1)  # (nqueries,)
            else :
                per_img_scores, _ = per_img_pred_cls[:, :, :-1].sigmoid().max(dim=-1)  # (nclips, nqueries)
                video_scores, labels = per_img_pred_cls[:, :, :-1].mean(dim=0).sigmoid().max(dim=-1)  # (nqueries,)
            
            
            # Compute final scores
            if self.class_agnostic_inference:
                scores = per_img_object_scores
                video_scores = video_scores
            else :
                scores  = (per_img_scores * per_img_object_scores) ** 0.5  # (nclips, nqueries)
                video_scores = (video_scores * video_object_scores) ** 0.5  # (nqueries,)
            
            # Convert clip-level scores to frame-level scores
            if hasattr(self, 'inference_clip_len') and self.inference_clip_len > 1:
                clip_len = self.inference_clip_len
                num_clips = scores.shape[0]
                num_frames_expected = pred_masks.shape[1]  # Get actual number of frames from masks
                
                # Repeat each clip's scores by clip_len
                scores_per_frame = []
                for clip_idx in range(num_clips):
                    clip_scores = scores[clip_idx]  # (nqueries,)
                    # For the last clip, only repeat for remaining frames
                    if clip_idx == num_clips - 1:
                        remaining_frames = num_frames_expected - (clip_idx * clip_len)
                        for _ in range(remaining_frames):
                            scores_per_frame.append(clip_scores)
                    else:
                        for _ in range(clip_len):
                            scores_per_frame.append(clip_scores)
                
                scores_per_frame_tensor = torch.stack(scores_per_frame, dim=0)  # (nframes, nqueries)
                
            else:
                scores_per_frame_tensor = scores  # No conversion needed

            # Use video_scores (from class predictions) for topk selection to maintain original behavior
            topk_scores, topk_indices = video_scores.topk(50, sorted=False)
            out_scores_per_frame = []
            for idx in topk_indices:
                out_scores_per_frame.append(scores_per_frame_tensor[:, idx].tolist())
            labels_per_image = labels[topk_indices]
            
            # Sort captions with topk indices
            if self.mask_captioning:
                out_caps = [mask_cap_results[i] for i in topk_indices]
            elif self.dummy_captioning:
                # caption should be the category name
                out_caps = [self.category_names[label] for label in labels_per_image]
            else :
                out_caps = []
            
            pred_masks = pred_masks[topk_indices.to(pred_masks.device)]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.
            out_masks = [m for m in masks.cpu()]
            
            if self.dummy_class_prediction:
                labels_per_image = torch.zeros_like(labels_per_image)
            elif self.lvvis_zs_vidstg_class_filtering:
                # Filter and map LVVIS categories to VidSTG
                assert labels_per_image.max() < 1196, f"Label {labels_per_image.max()} exceeds LVVIS range (0-1195)"
                # Create mask of predictions to keep
                keep_mask = torch.tensor([label.item() in self.lvvis_to_vidstg_mapping for label in labels_per_image], device=labels_per_image.device)
                # Map labels for kept predictions
                mapped_labels = torch.tensor([self.lvvis_to_vidstg_mapping[label.item()] for label in labels_per_image[keep_mask]], device=labels_per_image.device)
                # Filter all predictions
                labels_per_image = mapped_labels
                topk_scores = topk_scores[keep_mask]
                out_masks = [mask for mask, keep in zip(out_masks, keep_mask.cpu().tolist()) if keep]
                out_caps = [cap for cap, keep in zip(out_caps, keep_mask.cpu().tolist()) if keep]
                # Filter per-frame scores
                out_scores_per_frame = [scores for scores, keep in zip(out_scores_per_frame, keep_mask.cpu().tolist()) if keep]

            out_scores = topk_scores.tolist()
            out_labels = labels_per_image.tolist()
            
        else:
            out_scores = []
            out_labels = []
            out_caps = []
            out_masks = []
            out_boxes = []
        
        if not self.box_mode_on:
            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_masks": out_masks,

                "pred_scores_per_frame": out_scores_per_frame,
            }
        else :
            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_boxes": out_boxes,
            }
        if self.mask_captioning or self.dummy_captioning:
            video_output["pred_captions"] = out_caps

        video_output = self._apply_hysteresis_filter(video_output)
        video_output = self._apply_per_frame_score_threshold(video_output)
        return video_output

    def inference_video_dvoc(self, per_img_pred_cls, pred_cls, per_img_pred_obj, pred_obj, pred_masks_or_boxes, mask_cap_results, img_size, output_height, output_width):
        if len(pred_cls) > 0:

            frame_selec=True
            if frame_selec:
                # Stack predictions into tensors
                per_img_pred_cls = torch.stack(per_img_pred_cls, dim=0)  # (nclips, nqueries, nclasses)
                per_img_pred_obj = torch.stack(per_img_pred_obj, dim=0)  # (nclips, nqueries, 2)

                # Compute object and class scores
                per_img_object_scores = F.softmax(per_img_pred_obj, dim=-1)[:, :, 0]  # (nclips, nqueries)
                if self.filter_person_only:
                    # Filter out all classes except person (score=0)
                    per_img_pred_cls = per_img_pred_cls[:, :, 772:773]  # (nclips, nqueries, 1)
                    per_img_scores, _ = per_img_pred_cls[:, :, :].sigmoid().max(dim=-1) # (nclips, nqueries)
                    video_scores, labels = per_img_pred_cls[:, :, :].mean(dim=0).sigmoid().max(dim=-1)  # (nqueries,)
                else :
                    per_img_scores, _ = per_img_pred_cls[:, :, :-1].sigmoid().max(dim=-1)  # (nclips, nqueries)
                    video_scores, labels = per_img_pred_cls[:, :, :-1].mean(dim=0).sigmoid().max(dim=-1)  # (nqueries,)
                
                # Compute final scores
                if self.class_agnostic_inference:
                    scores = per_img_object_scores
                else :
                    scores = (per_img_scores * per_img_object_scores) ** 0.5  # (nclips, nqueries)
                
                # Convert clip-level scores to frame-level scores
                if hasattr(self, 'inference_clip_len') and self.inference_clip_len > 1:
                    clip_len = self.inference_clip_len
                    num_clips = scores.shape[0]  # (nclips, nqueries)
                    num_frames_expected = pred_masks_or_boxes.shape[1] if len(pred_masks_or_boxes.shape) > 1 else pred_masks_or_boxes.shape[0]
                    
                    # Repeat each clip's scores by clip_len
                    scores_per_frame = []
                    for clip_idx in range(num_clips):
                        clip_scores = scores[clip_idx]  # (nqueries,)
                        # For the last clip, only repeat for remaining frames
                        if clip_idx == num_clips - 1:
                            remaining_frames = num_frames_expected - (clip_idx * clip_len)
                            for _ in range(remaining_frames):
                                scores_per_frame.append(clip_scores)
                        else:
                            for _ in range(clip_len):
                                scores_per_frame.append(clip_scores)
                    
                    scores_per_frame_tensor = torch.stack(scores_per_frame, dim=0)  # (nframes, nqueries)
                    # Keep the clip-level scores for thresholding, but store frame-level for output
                    scores_frames = scores_per_frame_tensor
                else:
                    scores_frames = scores  # No conversion needed
                
                hysteresis = False
                ######
                # hysteresis = True
                temporal_filter = True
                nms = False
                if hysteresis:
                    from utils.hysteresis_filtering import bidirectional_hysteresis_batch
                    print("scores : ", scores)
                    print("scores shape:", scores.shape)
                    
                    t_min = 3 if temporal_filter else 1
                    keep = bidirectional_hysteresis_batch(scores.transpose(0,1), low_thresh=0.4, high_thresh=0.6, min_duration=t_min)
                    print("scores shape:", keep.shape)
                    print("scores :", keep)
                    keep_query = keep.any(dim=1)  # (nqueries,)
                    print("num kept queries :", keep_query.sum())


                    pred_masks_or_boxes = pred_masks_or_boxes[keep_query]
                    print("pred_masks_or_boxes shape:", pred_masks_or_boxes.shape)
                    out_boxes = []
                    c=self.inference_clip_len # clip length
                    num_total_frames = pred_masks_or_boxes.shape[1]
                    for q in range(pred_masks_or_boxes.shape[0]): # for each kept query
                        cur_boxes=[]
                        for f in range(per_img_scores.shape[0]): # for each clip
                            if keep[q,f]: #if clip selected
                                for x in pred_masks_or_boxes[q, f*c:(f+1)*c]: # add the c bboxes
                                    cur_boxes.append(x.round().to(torch.int).tolist())
                            else:
                                for _ in pred_masks_or_boxes[q, f*c:(f+1)*c]:
                                    cur_boxes.append(None)
                        assert len(cur_boxes) == num_total_frames , f"Expected {num_total_frames} boxes, got {len(cur_boxes)}"
                        out_boxes.append(cur_boxes)
                    
                    print("len out_boxes:", len(out_boxes))
                    print("len out_boxes[0]:", len(out_boxes[0]))
                    raise NotImplementedError("Stop")

                # Apply presence threshold
                presence_threshold = 0.2
                keep = scores > presence_threshold  # (nclips, nqueries)
                keep_query = keep.any(dim=0)  # (nqueries,)

                # Filtered output scores
                out_scores = video_scores[keep_query]  # (num_selected_queries,)
                out_labels = labels[keep_query]  # (num_selected_queries,)

                # Check pred_masks_or_boxes shape
                # print(f"Original pred_boxes shape: {pred_masks_or_boxes.shape}")

                # if pred_masks_or_boxes.shape[0] == 100 and pred_masks_or_boxes.shape[1] == 960:
                #     print("Reshaping pred_masks_or_boxes...")
                #     # Assuming we need to swap axes to match (nframes, nqueries, 4)

                # # Now ensure correct shape
                # assert pred_masks_or_boxes.shape[0] == 192, f"Expected first dim to be 192, got {pred_masks_or_boxes.shape[0]}"
                # assert pred_masks_or_boxes.shape[1] == 100, f"Expected second dim to be 100, got {pred_masks_or_boxes.shape[1]}"

                if not self.box_mode_on:
                    pred_masks = pred_masks_or_boxes[keep_query]

                    pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
                    pred_masks = F.interpolate(
                        pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
                    )

                    masks = pred_masks > 0.
                    out_masks = [m for m in masks.cpu()]
                    raise NotImplementedError("Stop")

                # Denormalize boxes
                pred_masks_or_boxes[:, :, [0, 2]] *= output_width 
                pred_masks_or_boxes[:, :, [1, 3]] *= output_height
    
                if self.box_xyxy:
                    # Convert boxes to xyxy
                    from utils.box_ops import box_xywh_to_xyxy
                    pred_masks_or_boxes = torch.cat([box_xywh_to_xyxy(x).unsqueeze(0) for x in pred_masks_or_boxes], dim=0)
                
                pred_masks_or_boxes = pred_masks_or_boxes.permute(1, 0, 2)  # Swap (nqueries, nframe, 4) → (nframe, nqueries, 4)

                # Process bounding boxes per query per clip
                c=self.inference_clip_len
                num_total_frames = pred_masks_or_boxes.shape[0]
            
                out_boxes = []
                out_scores_per_frame = []
                for q in range(keep_query.shape[0]):
                    if keep_query[q]:
                        cur_boxes = []
                        cur_scores = []
                        for f in range(per_img_pred_cls.shape[0]): 
                            if keep[f, q]: #for each clip, if clip selected
                                for x in pred_masks_or_boxes[f*c:(f+1)*c, q]: # add the c bboxes
                                    cur_boxes.append(x.round().to(torch.int).tolist())
                            else:
                                for _ in pred_masks_or_boxes[f*c:(f+1)*c, q]:
                                    cur_boxes.append(None)
                        # Use frame-level scores if available
                        if hasattr(self, 'inference_clip_len') and self.inference_clip_len > 1 and 'scores_frames' in locals():
                            # Extract per-frame scores for this query
                            cur_scores = [scores_frames[frame_idx, q].item() for frame_idx in range(num_total_frames)]
                        else:
                            # Fallback to repeating clip scores
                            for f in range(per_img_pred_cls.shape[0]):
                                for _ in pred_masks_or_boxes[f*c:(f+1)*c, q]:
                                    cur_scores.append(scores[f, q].item())
                        # assert and print if error
                        assert len(cur_boxes) == num_total_frames , f"Expected {num_total_frames} boxes, got {len(cur_boxes)}"
                        out_boxes.append(cur_boxes)

                        assert len(cur_scores) == num_total_frames , f"Expected {num_total_frames} scores, got {len(cur_scores)}"
                        out_scores_per_frame.append(cur_scores)
                
                # Create captions before filtering
                if mask_cap_results:
                    out_caps = [cap for cap, keep in zip(mask_cap_results, keep_query) if keep]
                elif self.dummy_captioning:
                    out_caps = [self.category_names[label] for label in out_labels]
                else:
                    out_caps = ["just testing"] * len(out_boxes)
                
                out_scores = out_scores.tolist()
                if self.dummy_class_prediction:
                    out_labels = torch.zeros_like(out_labels)
                elif self.lvvis_zs_vidstg_class_filtering:
                    # Filter and map LVVIS categories to VidSTG
                    assert out_labels.max() < 1196, f"Label {out_labels.max()} exceeds LVVIS range (0-1195)"
                    # Create mask of predictions to keep
                    keep_mask = torch.tensor([label.item() in self.lvvis_to_vidstg_mapping for label in out_labels], device=out_labels.device)
                    # Map labels for kept predictions
                    mapped_labels = torch.tensor([self.lvvis_to_vidstg_mapping[label.item()] for label in out_labels[keep_mask]], device=out_labels.device)
                    # Filter all predictions
                    out_labels = mapped_labels
                    out_scores = [score for score, keep in zip(out_scores, keep_mask.cpu().tolist()) if keep]
                    out_boxes = [box for box, keep in zip(out_boxes, keep_mask.cpu().tolist()) if keep]
                    out_scores_per_frame = [scores for scores, keep in zip(out_scores_per_frame, keep_mask.cpu().tolist()) if keep]
                    out_caps = [cap for cap, keep in zip(out_caps, keep_mask.cpu().tolist()) if keep]

                out_labels = out_labels.tolist()

            else :

                object_scores = F.softmax(pred_obj, dim=-1)[:,0] # shape (nqueries,)
                if self.dvoc_class_agnostic:
                    scores = torch.ones_like(object_scores)
                    labels = torch.zeros_like(object_scores, dtype=torch.int64)
                else:
                    scores, labels = pred_cls[:, :-1].sigmoid().max(dim=-1)
                scores = (scores * object_scores) ** 0.5

                keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
                out_scores = scores[keep]
                out_labels = labels[keep]
                if self.dummy_captioning:
                    out_caps = [self.category_names[label] for label in out_labels]
                    print("out_caps:", out_caps)
                    print("len category names:", len(self.category_names))
                    raise NotImplementedError("inference video dvoc")
                else :
                    out_caps = [mask_cap_results[i] for i in keep if i]
                
                # print("keep shape:", keep.shape)
                print("keep:", keep)
                # print("cur_scores :", out_scores)
                # print("cur_classes :", out_labels)
                # print("cur_cap :", out_caps)
                
                if not self.box_mode_on:
                    mask_pred= pred_masks_or_boxes.sigmoid()
                    cur_masks = mask_pred[keep]
                    cur_mask_cls = pred_cls[keep]
                    cur_mask_cls = cur_mask_cls[:, :-1]

                    cur_prob_masks = out_scores.view(-1, 1, 1) * cur_masks
                    print("mask_pred shape:", mask_pred.shape)
                    print("cur_masks shape:", cur_masks.shape)
                    print("cur_mask_cls shape:", cur_mask_cls.shape)
                    print("cur_prob_masks shape:", cur_prob_masks.shape)
                    raise NotImplementedError("Stop")
                
                if not self.box_mode_on:

                    pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
                    # print("pred_masks:", pred_masks.shape)
                    pred_masks = F.interpolate(
                        pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
                    )
                    # print("final shape:", pred_masks.shape)
                    # raise Exception("Check pred_masks")

                    masks = pred_masks > 0.
                    out_masks = [m for m in masks.cpu()]
                else :
                    pred_boxes = pred_masks_or_boxes[keep]
                    
                    # Denormalize boxes
                    pred_boxes[:, :, [0, 2]] *= output_width 
                    pred_boxes[:, :, [1, 3]] *= output_height
        
                    if self.box_xyxy:
                        # Convert boxes to xyxy
                        from utils.box_ops import box_xywh_to_xyxy
                        pred_boxes = torch.cat([box_xywh_to_xyxy(x).unsqueeze(0) for x in pred_boxes], dim=0)

                    # Round to int
                    # out_boxes = [m for m in pred_boxes.cpu()]
                    out_boxes = [m.round().to(torch.int) for m in pred_boxes.cpu()]
                out_scores = out_scores.tolist()
                out_labels = out_labels.tolist()
            



        else:
            out_scores = []
            out_labels = []
            out_caps = []
            out_masks = []
            out_boxes = []
        
        if not self.box_mode_on:
            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_captions": out_caps,
                "pred_masks": out_masks,
            }
        else :
            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_captions": out_caps,
                "pred_boxes": out_boxes,

                #temp
                "pred_scores_per_frame": out_scores_per_frame,
            }

        video_output = self._apply_hysteresis_filter(video_output)
        video_output = self._apply_per_frame_score_threshold(video_output)
        return video_output

    def _apply_per_frame_score_threshold(self, video_output):
        thr = self.per_frame_score_threshold
        if thr <= 0.0:
            return video_output
        if "pred_scores_per_frame" not in video_output:
            return video_output

        is_mask_mode = "pred_masks" in video_output and "pred_boxes" not in video_output

        if is_mask_mode:
            keep = [s > thr for s in video_output["pred_scores"]]
            if all(keep):
                return video_output
            out = {}
            for key, val in video_output.items():
                if key in ("pred_scores", "pred_labels", "pred_masks",
                           "pred_scores_per_frame", "pred_captions"):
                    out[key] = [v for v, k in zip(val, keep) if k]
                else:
                    out[key] = val
            return out

        # bbox-mode: per-frame nullification
        per_frame_key = "pred_boxes"
        if per_frame_key not in video_output:
            return video_output
        new_per_frame = []
        for track_scores, track_seq in zip(video_output["pred_scores_per_frame"], video_output[per_frame_key]):
            row = list(track_seq)
            T = min(len(track_scores), len(row))
            for t in range(T):
                if track_scores[t] <= thr:
                    row[t] = None
            new_per_frame.append(row)
        out = dict(video_output)
        out[per_frame_key] = new_per_frame
        return out

    def _apply_hysteresis_filter(self, video_output):
        """Filter per-track predictions via bidirectional hysteresis on the per-frame
        scores.
        """
        if not self.hysteresis_filter_enabled:
            return video_output
        if "pred_scores_per_frame" not in video_output:
            return video_output
        if len(video_output["pred_scores_per_frame"]) == 0:
            return video_output

        low, high, min_dur = self.hysteresis_t_low, self.hysteresis_t_high, self.hysteresis_min_duration
        scores = torch.tensor(video_output["pred_scores_per_frame"])  # (Nq, T)
        Nq, T = scores.shape

        # forward + backward + min-duration filter, per track
        active = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(Nq):
            fwd = torch.zeros(T, dtype=torch.bool)
            on = False
            for t in range(T):
                if scores[i, t] >= high: on = True
                elif scores[i, t] < low: on = False
                fwd[t] = on
            bwd = torch.zeros(T, dtype=torch.bool)
            on = False
            for t in range(T - 1, -1, -1):
                if scores[i, t] >= high: on = True
                elif scores[i, t] < low: on = False
                bwd[t] = on
            track_active = fwd | bwd
            # drop runs shorter than min_duration
            count = 0
            for t in range(T):
                if track_active[t]:
                    count += 1
                else:
                    if count < min_dur:
                        track_active[t - count: t] = False
                    count = 0
            active[i] = track_active

        kept = active.any(dim=1).tolist()
        is_mask_mode = "pred_masks" in video_output and "pred_boxes" not in video_output

        new_out = {"image_size": video_output["image_size"]}
        new_out["pred_scores"] = [s for s, k in zip(video_output["pred_scores"], kept) if k]
        new_out["pred_labels"] = [l for l, k in zip(video_output["pred_labels"], kept) if k]
        new_out["pred_scores_per_frame"] = [s for s, k in zip(video_output["pred_scores_per_frame"], kept) if k]
        if "pred_captions" in video_output:
            new_out["pred_captions"] = [c for c, k in zip(video_output["pred_captions"], kept) if k]

        if is_mask_mode:
            # mask-mode segmentations left untouched (track-level filtering)
            new_out["pred_masks"] = [m for m, k in zip(video_output["pred_masks"], kept) if k]
        else:
            per_frame_key = "pred_boxes" if "pred_boxes" in video_output else None
            if per_frame_key is not None:
                kept_per_frame = []
                for i, k in enumerate(kept):
                    if not k: continue
                    row = list(video_output[per_frame_key][i])
                    for t in range(min(T, len(row))):
                        if not active[i, t]:
                            row[t] = None
                    kept_per_frame.append(row)
                new_out[per_frame_key] = kept_per_frame
        return new_out

    def match_from_embds(self, tgt_embds, cur_embds):

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))  # (100,100)

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def match_from_embds_boxes_scores(self, tgt_embds, cur_embds, tgt_boxes, cur_boxes, tgt_cls_scores, src_cls_scores, cost_embd, cost_box, cost_cls, alpha=0.5):

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))  # (100,100)

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices
    
    def match_from_vote_embds(self, tgt_embds, cur_embds):
        cos_score_list=[]
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        indices_lists=[]
        cost_list=[]
        T = len(tgt_embds)
        top_t=self.topk_matching_num_k
        for i in range(len(tgt_embds)):
            tgt_embd = tgt_embds[i] / tgt_embds[i].norm(dim=1)[:, None]
            cos_sim = torch.mm(cur_embds, tgt_embd.transpose(0, 1))
            cost_embd = 1 - cos_sim
            cost_embd = cost_embd.cpu()
            C = 1.0 * cost_embd
            # C = C.cpu()
            cos_score=C*(i+1)
            cos_score_list.append(cos_score)
            indice = linear_sum_assignment(C.transpose(0, 1))  
            indices_lists.append(indice[1]) 
            cost_score=cost_embd[indice[1],indice[0]].sum()
            cost_list.append(cost_score)
        costs=torch.stack(cost_list,dim=0)
        cos_scores=torch.stack(cos_score_list,dim=0)
        
        # cast to float32: torch.topk on CPU doesn't support fp16 until PyTorch 2.3
        costs = costs.float()
        if costs.shape[0]<top_t:
            min_cost_indice=torch.topk(costs,k=1,largest=False)
            indices=indices_lists[min_cost_indice[1]]
        else:
            min_cost_indice=torch.topk(costs,k=top_t,largest=False)
            c_score=cos_scores[min_cost_indice[1]].mean(0)
            final_indices = linear_sum_assignment(c_score.transpose(0, 1)) 
            indices=final_indices[1]
        return indices
    


    def get_association_matrix(self, mask_embed):
        """
        Compute the global dot-product similarity matrix from a list of per-frame embeddings.

        Args:
            mask_embed: list of length T, each item a tensor of shape [N, C]
                        (T frames, N objects per frame, C-dim features)

        Returns:
            asso_matrix: tensor of shape [T*N, T*N] with dot product similarities
        """
        # Concatenate all embeddings into one (T*N, C) tensor and normalize
        all_embeddings = torch.cat(mask_embed, dim=0)  # (T*N, C)
        all_embeddings = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
        # Compute dot-product similarity matrix
        association_mat=torch.mm(all_embeddings, all_embeddings.T) 
        #softmax 
        return F.softmax(association_mat, dim=-1)
        
        #Sigmoid (values between 0 and 1)
        # return F.sigmoid(association_mat)  
        
    def greedy_extract_trajectories(self, asso_scores, num_frames):
        """
        Greedily convert an association matrix to discrete tracking IDs.

        Args:
            asso_scores: Tensor of shape (num_tot_objs, num_tot_objs)
            num_frames: int
            thresh: float

        Returns:
            ids: Tensor of shape (num_tot_objs,)
        """
        assert asso_scores.min()>0, "asso_scores min is less than 0"
        assert asso_scores.max()<=1, "asso_scores max is greater than 1"

        num_tot_objs = asso_scores.shape[0]
        assert num_frames > 0 and num_tot_objs % num_frames == 0
        num_objs = num_tot_objs // num_frames

        # Don't merge objects in the same frame
        idx = torch.arange(num_tot_objs)
        frame_ids = idx // num_objs
        mask = (frame_ids[:, None] != frame_ids[None, :])
        mask |= torch.eye(num_tot_objs, dtype=torch.bool)
        asso_scores = asso_scores * mask.to(self.device)

        ids = torch.zeros(num_tot_objs, dtype=torch.int32, device=self.device)
        i=0
        while ids.min() == 0 :
            i+=1
            scores_3d = asso_scores.reshape(num_tot_objs, num_frames, num_objs)
            best_per_frame, best_id = scores_3d.max(dim=2)  # (num_tot_objs, num_frames)
            traj_scores = best_per_frame.sum(dim=1)       # (num_tot_objs,)

            ind = torch.argmax(traj_scores)
            id_count = ids.max() + 1

            # Only keep the max in each frame
            merge_inds = torch.nn.functional.one_hot(best_id[ind], num_objs).reshape(-1).bool()

            ids += merge_inds.to(torch.int32) * id_count
            
            # Zero out rows and columns of merged inds
            outer = torch.ger(1. - merge_inds.float(), 1. - merge_inds.float())
            asso_scores = asso_scores * outer
        ids=(ids-1).reshape(num_frames,num_objs) # (ids from 0 to N-1)
        return ids.cpu().numpy()

    def og_greedy_extract_trajectories(self, asso_scores, num_frames, thresh=0.3):
        """
        Greedily convert an association matrix to discrete tracking IDs.

        Args:
            asso_scores: Tensor of shape (num_tot_objs, num_tot_objs)
            num_frames: int
            thresh: float

        Returns:
            ids: Tensor of shape (num_tot_objs,)
        """

        num_tot_objs = asso_scores.shape[0]
        assert num_frames > 0 and num_tot_objs % num_frames == 0
        num_objs = num_tot_objs // num_frames

        # Don't merge objects in the same frame
        idx = torch.arange(num_tot_objs)
        frame_ids = idx // num_objs
        mask = (frame_ids[:, None] != frame_ids[None, :])
        mask |= torch.eye(num_tot_objs, dtype=torch.bool)
        asso_scores = asso_scores * mask.to(self.device)

        ids = torch.zeros(num_tot_objs, dtype=torch.int32, device=self.device)
        i=0
        while asso_scores.max() >= thresh:
            i+=1
            print("round :", i)
            can_merge = asso_scores >= thresh  # (num_tot_objs, num_tot_objs)
            num_merges = can_merge.sum(dim=1)
            print("num_merges :",num_merges)
            ind = torch.argmax(num_merges)
            print("ind :",ind)
            id_count = ids.max() + 1
            print("id_count :",id_count)
            merge_inds = can_merge[ind].clone()

            # Only keep the max in each frame
            scores = asso_scores[ind].reshape(num_frames, num_objs)
            max_ind_in_frame = scores.argmax(dim=1)
            is_max_score = torch.nn.functional.one_hot(max_ind_in_frame, num_objs).bool()
            is_max_score = is_max_score.reshape(-1)
            merge_inds &= is_max_score

            ids += merge_inds.to(torch.int32) * id_count
            print("ids :", ids.reshape(num_frames, num_objs))
            print("\n")
            # Zero out rows and columns of merged inds
            outer = torch.ger(1. - merge_inds.float(), 1. - merge_inds.float())
            asso_scores = asso_scores * outer
        ids=(ids-1).reshape(num_frames,num_objs) # (ids from 0 to N-1)
        return ids.cpu().numpy()
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_pred_score(self, pred_cls, pred_obj):
        scores_cls = pred_cls[:, :-1].sigmoid()
        object_scores = F.softmax(pred_obj, dim=-1)[:, :-1]
        scores = (scores_cls * object_scores) ** 0.5

        if self.filter_person_only:
            # Filter out all classes except person (score=0)
            scores=scores[:, 772:773]
        return scores, scores_cls, object_scores
    
    def compute_timing_statistics(self):
        """
        Compute timing statistics from accumulated data.
        Returns a dictionary with average latencies and FPS for each component.
        """
        if not self.timing_stats['detection_times']:
            logger.warning("No timing data available. Set enable_timing=True during inference.")
            return {}
        
        import numpy as np
        
        detection_times = np.array(self.timing_stats['detection_times'])
        tracking_times = np.array(self.timing_stats['tracking_times'])
        captioning_times = np.array(self.timing_stats['captioning_times'])
        post_processing_times = np.array(self.timing_stats['post_processing_times'])
        total_times = np.array(self.timing_stats['total_times'])
        num_frames = np.array(self.timing_stats['num_frames_per_video'])
        
        # Compute averages per video
        stats = {
            'per_video_avg': {
                'detection_ms': float(np.mean(detection_times)),
                'tracking_ms': float(np.mean(tracking_times)),
                'captioning_ms': float(np.mean(captioning_times)),
                'post_processing_ms': float(np.mean(post_processing_times)),
                'total_ms': float(np.mean(total_times)),
                'detection_fps': float(np.mean(num_frames / (detection_times / 1000))),
                'tracking_fps': float(np.mean(num_frames / (tracking_times / 1000))),
                'captioning_fps': float(np.mean(num_frames / (captioning_times / 1000))),
                'post_processing_fps': float(np.mean(num_frames / (post_processing_times / 1000))),
                'total_fps': float(np.mean(num_frames / (total_times / 1000))),
            },
            'dataset_total': {
                'num_videos': len(detection_times),
                'total_frames': int(np.sum(num_frames)),
                'detection_ms_total': float(np.sum(detection_times)),
                'tracking_ms_total': float(np.sum(tracking_times)),
                'captioning_ms_total': float(np.sum(captioning_times)),
                'post_processing_ms_total': float(np.sum(post_processing_times)),
                'total_ms_total': float(np.sum(total_times)),
                'detection_fps_overall': float(np.sum(num_frames) / (np.sum(detection_times) / 1000)),
                'tracking_fps_overall': float(np.sum(num_frames) / (np.sum(tracking_times) / 1000)),
                'captioning_fps_overall': float(np.sum(num_frames) / (np.sum(captioning_times) / 1000)),
                'post_processing_fps_overall': float(np.sum(num_frames) / (np.sum(post_processing_times) / 1000)),
                'total_fps_overall': float(np.sum(num_frames) / (np.sum(total_times) / 1000)),
            },
            'per_video_std': {
                'detection_ms': float(np.std(detection_times)),
                'tracking_ms': float(np.std(tracking_times)),
                'captioning_ms': float(np.std(captioning_times)),
                'post_processing_ms': float(np.std(post_processing_times)),
                'total_ms': float(np.std(total_times)),
            },
            'per_video_stats': self.timing_stats['per_video_stats']
        }
        
        return stats
    
    def save_timing_statistics(self, output_path='timing_statistics.json'):
        """
        Save timing statistics to JSON files.
        Summary stats (averages, totals, std) saved to output_path.
        Detailed per-video stats saved to output_path with '_detail' suffix.
        
        Args:
            output_path: Path to save the summary statistics JSON file
        """
        stats = self.compute_timing_statistics()
        
        if not stats:
            logger.warning("No statistics to save.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Split into summary and detail
        summary_stats = {
            'per_video_avg': stats['per_video_avg'],
            'dataset_total': stats['dataset_total'],
            'per_video_std': stats['per_video_std']
        }
        
        detail_stats = {
            'per_video_stats': stats['per_video_stats']
        }
        
        # Save summary file
        with open(output_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Timing summary saved to: {output_path}")
        
        # Save detail file
        detail_path = output_path.replace('.json', '_detail.json')
        with open(detail_path, 'w') as f:
            json.dump(detail_stats, f, indent=2)
        logger.info(f"Timing details saved to: {detail_path}")
        
        logger.info(f"Summary (avg per video):")
        logger.info(f"  Detection: {stats['per_video_avg']['detection_ms']:.2f} ms ({stats['per_video_avg']['detection_fps']:.2f} FPS)")
        logger.info(f"  Tracking: {stats['per_video_avg']['tracking_ms']:.2f} ms ({stats['per_video_avg']['tracking_fps']:.2f} FPS)")
        logger.info(f"  Captioning: {stats['per_video_avg']['captioning_ms']:.2f} ms ({stats['per_video_avg']['captioning_fps']:.2f} FPS)")
        logger.info(f"  Post-processing: {stats['per_video_avg']['post_processing_ms']:.2f} ms ({stats['per_video_avg']['post_processing_fps']:.2f} FPS)")
        logger.info(f"  Total: {stats['per_video_avg']['total_ms']:.2f} ms ({stats['per_video_avg']['total_fps']:.2f} FPS)")
        
        return stats
    
    def reset_timing_statistics(self):
        """Reset all accumulated timing statistics."""
        self.timing_stats = {
            'detection_times': [],
            'tracking_times': [],
            'captioning_times': [],
            'post_processing_times': [],
            'total_times': [],
            'num_frames_per_video': [],
            'per_video_stats': [],
        }
        self.timing_video_count = 0