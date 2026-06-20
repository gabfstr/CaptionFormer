# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer/ovformer_model.py
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.captioning_head import build_captioning_head


@META_ARCH_REGISTRY.register()
class CaptionFormer(nn.Module):
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
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        agnostic_classifier: bool,
        clip_classifier: bool,
        clip_image_path: str,
        # Box instead of mask 
        use_masks: bool,
        use_boxes: bool,
        box_mode_on: bool,
        box_xyxy: bool,
        dvoc_inference: bool,
        vidstg_img: bool,

        # freeze all but captioning head
        tune_captioning_head: bool,

        ## Experimental
        debugging_mode_on: bool,
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
        
        #### Experimental
        self.debugging_mode_on=debugging_mode_on
        
        
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        
        self.mask_captioning = mask_captioning
        if self.mask_captioning:
            self.captioning_head = captioning_head
        self.tune_captioning_head = tune_captioning_head
        
        self.use_masks = use_masks
        self.use_boxes = use_boxes

        self.box_mode_on = box_mode_on
        self.box_xyxy = box_xyxy
        self.dvoc_inference = dvoc_inference

        self.vidstg_img=vidstg_img

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

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.agnostic_classifier = agnostic_classifier
        
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
        self.clip_classifier = clip_classifier
        if self.clip_classifier:
            self.clip = torch.load(clip_image_path, map_location=self.device)

        self.i_debug = 0
        if self.debugging_mode_on:
            self.i_debug=0


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
            captioning_head = build_captioning_head(cfg, cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
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

        mask_loss = cfg.MODEL.MASK_FORMER.USE_MASKS
        box_loss = cfg.MODEL.MASK_FORMER.USE_BOXES
        if not mask_loss :
            dice_weight = 0.0
            mask_weight = 0.0
        if not box_loss :
            box_l1_weight = 0.0
            box_giou_weight = 0.0
        
        # building criterion
        matcher = HungarianMatcher(
            cost_object=object_weight,
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_bbox=box_l1_weight,
            cost_giou=box_giou_weight,
            by='mask' if not box_loss else 'box', 
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_object_ce": object_weight, "loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, 
                       "loss_caption": caption_weight, "loss_l1_box": box_l1_weight, "loss_giou_box": box_giou_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # Caption loss is NOT in criterion
        losses=["labels"]
        if box_loss:
            losses += ["boxes"]
        if mask_loss:
            losses += ["masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            empty_object_coeff=cfg.MODEL.MASK_FORMER.EMPTY_OBJECT_WEIGHT
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "mask_captioning": mask_captioning,
            "captioning_head": captioning_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "agnostic_classifier": cfg.MODEL.MASK_FORMER.AGNOSTIC_CLASSIFIER,
            "clip_classifier": cfg.MODEL.MASK_FORMER.CLIP_CLASSIFIER,
            "clip_image_path": cfg.MODEL.MASK_FORMER.CLIP_IMAGE_PATH,
            "use_masks": cfg.MODEL.MASK_FORMER.USE_MASKS,
            "use_boxes": cfg.MODEL.MASK_FORMER.USE_BOXES,
            "box_mode_on": cfg.MODEL.MASK_FORMER.BOX_MODE_ON,
            "vidstg_img": ("vidstg" in cfg.DATASETS.TRAIN[0]) or ("vidstg" in cfg.DATASETS.TEST[0]),
            "box_xyxy": cfg.MODEL.MASK_FORMER.BOX_XYXY,
            "dvoc_inference": cfg.MODEL.MASK_FORMER.DVOC_INFERENCE,

            "tune_captioning_head": cfg.MODEL.MASK_FORMER.TUNE_CAPTIONING_HEAD,

            ### Expermiental
            "debugging_mode_on": cfg.MODEL.MASK_FORMER.DEBUGGING_MODE_ON,
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
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        if self.debugging_mode_on :
            #### Check Image & gt
            index_vis=0
            print("Image ID:", batched_inputs[index_vis]['image_id'])
            print("Dataset:", batched_inputs[index_vis]['dataset'])
            dataset_name = batched_inputs[index_vis]['dataset']
            print("keys in batched_inputs:", batched_inputs[index_vis].keys())
            print("Original image shape", batched_inputs[index_vis]['width'], batched_inputs[index_vis]['height'])
            print("True image shape:", batched_inputs[index_vis]['image'].shape)
            print("image shape:", images[index_vis].shape)
            if 'instances' in batched_inputs[index_vis] and len(batched_inputs[index_vis]['instances']) > 0:
                if hasattr(batched_inputs[index_vis]['instances'], "gt_masks"):
                    print("gt_masks shape:", batched_inputs[index_vis]['instances'].gt_masks.shape)
                    gt_msk = batched_inputs[index_vis]['instances'].gt_masks        
                    gt_msk = gt_msk.cpu()
                gt_boxes = batched_inputs[index_vis]['instances'].gt_boxes
                gt_boxes = gt_boxes.tensor.cpu()
                img_id = batched_inputs[index_vis]['image_id']
                print("image_id:", img_id)
                print("nb instances:", len(gt_boxes))

                
                # CHECK IMAGE INPUT ??
                from .utils.visualization import Segmentation
                filename='./TEST_MASK2FORMER_INPUT.png'
                visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                if len(gt_boxes) == 0:
                    # vis image
                    visualization.plot_image(images[index_vis].unsqueeze(0).cpu(), filename)
                    raise Exception("stop here")
                if not self.box_mode_on:
                    # # mask
                    visualization(images[index_vis].unsqueeze(0).cpu(), gt_msk.unsqueeze(0).cpu(), filename)
                else :
                    # box
                    visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), gt_boxes.cpu(), filename)

        features_clip = []
        if self.clip_classifier:
            for i in range(len(batched_inputs)):
                if self.vidstg_img:
                    feature_clip = self.clip[batched_inputs[i]['file_name'].replace('./','')].to(self.device).to(torch.float32)
                else :
                    feature_clip = self.clip[batched_inputs[i]['image_id']].to(self.device).to(torch.float32)
                features_clip.append(feature_clip)

        features = self.backbone(images.tensor)

        # Resize images to bs, 364, 364 for ViT
        # vit_images = F.interpolate(images.tensor, size=(364, 364), mode='bilinear', align_corners=False)
        outputs = self.sem_seg_head(features, features_clip)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                datasets = [x["dataset"] for x in batched_inputs]                
                targets = self.prepare_targets(gt_instances, images, datasets)
            else:
                targets = None
            
            losses, indices = self.criterion(outputs, targets)
            
            if self.debugging_mode_on :
                
                from fvcore.nn import smooth_l1_loss
                from utils.box_ops import generalized_box_iou

                def _get_src_permutation_idx( indices):
                    # permute predictions following indices
                    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
                    src_idx = torch.cat([src for (src, _) in indices])
                    return batch_idx, src_idx

                def _get_tgt_permutation_idx( indices):
                    # permute targets following indices
                    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
                    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
                    return batch_idx, tgt_idx
                def _permute_tgt_boxes(tgt_boxes, tgt_idx:tuple):
                    """
                    Efficiently reorder target boxes based on indexing tensors.
                    Args:
                        tgt_boxes (list of torch.Tensor): List of tensors, where each tensor contains bounding boxes
                                                        for one image. Each tensor has shape (num_boxes, 4).
                        tgt_idx (tuple of torch.Tensor): Tuple of two tensors:
                            - First tensor: indices specifying which image to sample from.
                            - Second tensor: indices specifying which box to sample from the chosen image.

                    Returns:
                        torch.Tensor: A reordered tensor of shape (len(tgt_idx[0]), 4), containing the selected boxes.
                    """
                    # Concatenate all boxes into a single tensor
                    all_boxes = torch.cat(tgt_boxes, dim=0)
                    # Compute offsets for each image
                    offsets = torch.cumsum(torch.tensor([0] + [len(boxes) for boxes in tgt_boxes[:-1]]), dim=0)
                    # Adjust box indices with offsets to create a single flattened index
                    flat_indices = offsets[tgt_idx[0]] + tgt_idx[1]
                    # Use the flattened indices to gather the boxes
                    ordered_gt_boxes = all_boxes[flat_indices]

                    return ordered_gt_boxes
                print("indices:", indices)
                src_idx = _get_src_permutation_idx(indices)
                tgt_idx = _get_tgt_permutation_idx(indices)

                # Object logits
                src_logits = outputs["pred_logits"].clone().float()
                src_object_logits = outputs["pred_object_logits"].clone().float()
                
                idx = _get_src_permutation_idx(indices)
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(
                    src_logits.shape[:2], self.sem_seg_head.num_classes, dtype=torch.int64, device=src_logits.device
                )
                target_classes[idx] = target_classes_o
                target_object_classes = (target_classes == self.sem_seg_head.num_classes).long()
                
                print("targt_classes:", target_classes)
                print("shape target_classes:", target_classes.shape)
                print("target_object_classes:", target_object_classes)
                print("shape target_object_classes:", target_object_classes.shape)

                print("src_logits:", src_logits)
                print("shape src_logits:", src_logits.shape)
                print("src_object_logits:", src_object_logits)
                print("shape src_object_logits:", src_object_logits.shape)
                
                empty_object_weight = torch.ones(2, device = target_object_classes.device)
                empty_object_weight[-1] = 0.4


                if self.box_mode_on:
                    src_boxes = outputs["pred_boxes"].clone().float()
                    src_boxes_xywh = src_boxes[src_idx]
                    target_boxes = [t["boxes"].clone().float() for t in targets]
                    target_boxes_xywh = _permute_tgt_boxes(target_boxes, tgt_idx)

                    # Convert from xywh to xyxy
                    src_boxes_xyxy = torch.cat([src_boxes_xywh[:, :2] , torch.max(src_boxes_xywh[:, :2], src_boxes_xywh[:, :2] + src_boxes_xywh[:, 2:])], dim=1)
                    target_boxes_xyxy = torch.cat([target_boxes_xywh[:, :2] , torch.max(target_boxes_xywh[:, :2], target_boxes_xywh[:, :2] + target_boxes_xywh[:, 2:])], dim=1)
                    

                

                    fake_losses = {
                        "loss_giou_box": 1 - generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy).mean() if target_boxes_xyxy.numel() > 0 else torch.tensor(0.0, device=target_boxes_xywh.device),
                        "loss_l1_box": smooth_l1_loss(src_boxes_xywh, target_boxes_xywh, beta=0.0, reduction="mean"),
                        "loss_object_ce" : F.cross_entropy(src_object_logits.transpose(1, 2), target_object_classes, empty_object_weight),
                    }

                    print("losses : ", fake_losses)
                    
                    h_pad, w_pad = images.tensor.shape[-2:]
                    print("og boxes : ", target_boxes_xywh)
                    print("h_pad, w_pad : ", h_pad, w_pad)
                    print("boxes after : ", target_boxes_xywh * torch.tensor([w_pad, h_pad, w_pad, h_pad], device=target_boxes_xywh.device))
                    print("src boxes og : ", src_boxes_xywh)
                    print("src boxes after : ", src_boxes_xywh * torch.tensor([w_pad, h_pad, w_pad, h_pad], device=src_boxes_xywh.device))
                    # Denormalize boxes
                    target_boxes_xywh[:, [0, 2]] *= w_pad
                    target_boxes_xywh[:, [1, 3]] *= h_pad  
                    src_boxes_xywh[:, [0, 2]] *= w_pad
                    src_boxes_xywh[:, [1, 3]] *= h_pad

                    # CHECK IMAGE INPUT ??
                    from .utils.visualization import Segmentation
                    filename='./TEST_MASK2FORMER_GT.png'
                    
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # box
                    visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), target_boxes_xywh.cpu(), filename)

                    filename='./TEST_MASK2FORMER_PRED.png'
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # box
                    visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), src_boxes_xywh.cpu(), filename)
                    print("Saved prediction at :", filename)
                    print("\n\n")
                    # from time import sleep
                    # sleep(1)
                    self.i_debug += 1
        
                    if (self.i_debug > 6) and (len(gt_boxes)>0):
                        raise Exception("stop : debugging")
                    
                    if dataset_name=='visualgenome':
                        print("filename:", filename)
                        raise NotImplementedError("Visualize image[0] for visualgenome")
                    # if len(gt_boxes) > 0 :
                    #     raise Exception("stop : debugging")

                else :
                    from captionformer.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

                    from detectron2.projects.point_rend.point_features import (
                        get_uncertain_point_coords_with_randomness,
                        point_sample,
                    )

                    def dice_loss(
                            inputs: torch.Tensor,
                            targets: torch.Tensor,
                            num_masks: float,
                        ):
                        """
                        Compute the DICE loss, similar to generalized IOU for masks
                        Args:
                            inputs: A float tensor of arbitrary shape.
                                    The predictions for each example.
                            targets: A float tensor with the same shape as inputs. Stores the binary
                                    classification label for each element in inputs
                                    (0 for the negative class and 1 for the positive class).
                        """
                        inputs = inputs.sigmoid()
                        inputs = inputs.flatten(1)
                        numerator = 2 * (inputs * targets).sum(-1)
                        denominator = inputs.sum(-1) + targets.sum(-1)
                        loss = 1 - (numerator + 1) / (denominator + 1)
                        return loss.sum() / num_masks


                    dice_loss_jit = torch.jit.script(
                        dice_loss
                    )  # type: torch.jit.ScriptModule


                    def sigmoid_ce_loss(
                            inputs: torch.Tensor,
                            targets: torch.Tensor,
                            num_masks: float,
                        ):
                        """
                        Args:
                            inputs: A float tensor of arbitrary shape.
                                    The predictions for each example.
                            targets: A float tensor with the same shape as inputs. Stores the binary
                                    classification label for each element in inputs
                                    (0 for the negative class and 1 for the positive class).
                        Returns:
                            Loss tensor
                        """
                        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

                        return loss.mean(1).sum() / num_masks


                    sigmoid_ce_loss_jit = torch.jit.script(
                        sigmoid_ce_loss
                    )  # type: torch.jit.ScriptModule


                    def calculate_uncertainty(logits):
                        """
                        We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
                            foreground class in `classes`.
                        Args:
                            logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
                                class-agnostic, where R is the total number of predicted masks in all images and C is
                                the number of foreground classes. The values are logits.
                        Returns:
                            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
                                the most uncertain locations having the highest uncertainty score.
                        """
                        assert logits.shape[1] == 1
                        gt_class_logits = logits.clone()
                        return -(torch.abs(gt_class_logits))

                    
                    src_masks = outputs["pred_masks"].clone()
                    src_masks = src_masks[src_idx]
                    masks = [t["masks"].clone() for t in targets]
                    # TODO use valid to mask invalid areas due to padding in loss
                    target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
                    target_masks = target_masks.to(src_masks)
                    target_masks = target_masks[tgt_idx]

                    # No need to upsample predictions as we are using normalized coordinates :)
                    # N x 1 x H x W
                    src_masks = src_masks[:, None]
                    target_masks = target_masks[:, None]

                    with torch.no_grad():
                        # sample point_coords
                        point_coords = get_uncertain_point_coords_with_randomness(
                            src_masks,
                            lambda logits: calculate_uncertainty(logits),
                            self.criterion.num_points,
                            self.criterion.oversample_ratio,
                            self.criterion.importance_sample_ratio,
                        )
                        # get gt labels
                        point_labels = point_sample(
                            target_masks,
                            point_coords,
                            align_corners=False,
                        ).squeeze(1)

                    point_logits = point_sample(
                        src_masks,
                        point_coords,
                        align_corners=False,
                    ).squeeze(1)

                    num_masks = sum(len(t["labels"]) for t in targets)

                    fake_losses = {
                        "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
                        "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
                        "loss_object_ce" : F.cross_entropy(src_object_logits.transpose(1, 2), target_object_classes, empty_object_weight),
                    }
                        

                    print("losses : ", fake_losses)
                    print("true losses:", losses)
                    # CHECK IMAGE INPUT ??
                    from .utils.visualization import Segmentation
                    filename='./TEST_MASK2FORMER_GT.png'
                    print("target_masks.shape:", target_masks.shape)
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # # mask
                    visualization(images[index_vis].unsqueeze(0).cpu(), target_masks.transpose(0,1).cpu(), filename)

                    filename='./TEST_MASK2FORMER_PRED.png'
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # # # mask
                    print("src_masks.shape:", src_masks.shape)
                    src_masks = F.interpolate(src_masks, size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode="bilinear", align_corners=False)
                    src_masks = (src_masks > 0).float()
                    print("after interpolation src_masks.shape:", src_masks.shape)
                    print("type of src_masks:", src_masks.dtype)
                    print("sum of pixels in src_masks:", src_masks.sum())
                    visualization(images[index_vis].unsqueeze(0).cpu(), src_masks.transpose(0,1).cpu(), filename)
                    print("Saved prediction at :", filename)
                    print("\n\n")
                    # from time import sleep
                    # sleep(1)
                    self.i_debug += 1
                    if self.i_debug > 100 and num_masks >0:
                        raise Exception("stop : debugging")
                    # if num_masks >0:
                    #     raise Exception("stop : debugging")
            

            # Captioning loss
            if self.mask_captioning:
                pred_queries = outputs["pred_queries"]
                if self.use_masks :
                    pred_masks_or_boxes = outputs["pred_masks"]
                else :
                    pred_masks_or_boxes = outputs["pred_boxes"]
                feature_ids = [x["image_id"] for x in batched_inputs]
                caption_loss = self.captioning_head({"image": images.tensor, "target" : targets, "feature_id" : feature_ids }, pred_queries, pred_masks_or_boxes, indices)

                losses["loss_caption"] = caption_loss
                

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_object_cls_results = outputs["pred_object_logits"]
            mask_pred_results = outputs["pred_masks"]
            box_pred_results = outputs["pred_boxes"]

            # Captioning inference
            if self.mask_captioning:
                pred_queries = outputs["pred_queries"]
                caption_pred_results = self.captioning_head.generate({"image": images.tensor, "text_input" : ["a photo of"] * len(images.tensor)}, pred_queries, mask_pred_results)
                print("Captions :", caption_pred_results)
                print("len captions :", len(caption_pred_results))
                print("len of a single caption :", len(caption_pred_results[0]))

            else :
                caption_pred_results = None
            
            if self.box_mode_on:
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )


                del outputs


                processed_results = []
                for mask_cls_result, mask_object_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_object_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})
                    
                
                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_object_cls_results, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r
                if self.debugging_mode_on :
                    index_vis=0
                    img_id = batched_inputs[index_vis]['image_id']
                    print("image_id:", img_id)

                    # pred_masks = mask_pred_results.cpu()
                    pred_masks = processed_results[-1]["instances"].pred_masks.cpu()
                    labels = instance_r.pred_classes.cpu()
                    scores = instance_r.scores.cpu()
                    print("pred_masks :", pred_masks.shape)

                    print("labels :", labels)
                    print("scores :", scores)

                    # Filter predictions
                    thresh = 0.4 
                    keep = scores > thresh
                    pred_masks = pred_masks[keep]
                    labels = labels[keep]
                    scores = scores[keep]
                    print("pred_masks :", pred_masks.shape)

                    print("labels :", labels)
                    print("scores :", scores)
                
                    # CHECK IMAGE INPUT ??
                    from .utils.visualization import Segmentation
                    filename='./TEST_MASK2FORMER_PRED.png'
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # # mask
                    print("image shape:", images[index_vis].shape)
                    print("pred_masks :", pred_masks.shape)
                    print("sum of pixels in pred_masks:", pred_masks.sum())

                    # Sort by score
                    scores, indices = scores.sort(descending=True)
                    pred_masks = pred_masks[indices]
                    labels = labels[indices]

                    # Take top 2
                    top=10
                    pred_masks = pred_masks[:top]
                    labels = labels[:top]
                    scores = scores[:top]
                    print("scores :", scores)
                    print("labels :", labels)

                    if self.mask_captioning:
                        # captions = caption_pred_results[indices]
                        # captions = captions[:4]
                        print("captions :")
                        for idx in indices[:top]:
                            print("-",caption_pred_results[i])

                        # print("Captions :", captions)
                    visualization(images[index_vis].unsqueeze(0).cpu(), pred_masks.unsqueeze(0).cpu(), filename, contours=True)
                    # box
                    # visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), pred_boxes.cpu(), filename)
                    self.i_debug+=1
                    if self.i_debug>10:
                        raise NotImplementedError("Visualize image[0]")
            else :

                del outputs
                processed_results = []
                for mask_cls_result, box_pred_result, input_per_image, image_size in zip(mask_cls_results, box_pred_results, batched_inputs, images.image_sizes):
                    
                    height, width = image_size[0], image_size[1]
                    pad_h, pad_w = images.tensor.shape[-2:]

                    # Convert from relative coordinates [0,1] with padding to relative [0,1] without padding
                    ratio_h = pad_h / height
                    ratio_w = pad_w / width 
                    box_pred_result[:, [0, 2]] *= ratio_w
                    box_pred_result[:, [1, 3]] *= ratio_h
                    # Clamp values to (0,1)
                    box_pred_result[:, [0, 2]] = box_pred_result[:, [0, 2]].clamp(0, 1)
                    box_pred_result[:, [1, 3]] = box_pred_result[:, [1, 3]].clamp(0, 1)

                    # Upsample boxes
                    true_height = input_per_image.get("height", image_size[0])
                    true_width = input_per_image.get("width", image_size[1])
                    box_pred_result[:, [0, 2]] *= true_width
                    box_pred_result[:, [1, 3]] *= true_height
                    
                    if self.dvoc_inference :
                        instance_r = retry_if_cuda_oom(self.box_dvoc_inference)(mask_cls_result, mask_object_cls_results, box_pred_result, image_size=(images.tensor.shape[-2], images.tensor.shape[-1]))
                    else : 
                        instance_r = retry_if_cuda_oom(self.box_instance_inference)(mask_cls_result, mask_object_cls_results, box_pred_result, image_size=(images.tensor.shape[-2], images.tensor.shape[-1]))
                    processed_results.append({"instances": instance_r})

                if self.debugging_mode_on :
                    index_vis=0
                    img_id = batched_inputs[index_vis]['image_id']
                    print("image_id:", img_id)

                    out_height, out_width = images.tensor.shape[-2], images.tensor.shape[-1]
                    true_h, true_w = batched_inputs[index_vis]['height'], batched_inputs[index_vis]['width']

                    Ratio_h = out_height/true_h
                    Ratio_w = out_width/true_w

                    pred_boxes = instance_r.pred_boxes.tensor.cpu()
                    labels = instance_r.pred_classes.cpu()
                    scores = instance_r.scores.cpu()
                    
                    # get boxes to right scale
                    pred_boxes[:, [0, 2]] *= Ratio_w
                    pred_boxes[:, [1, 3]] *= Ratio_h

                    if self.box_xyxy:
                        from utils.box_ops import box_xyxy_to_xywh
                        pred_boxes = box_xyxy_to_xywh(pred_boxes)
                    print("pred_boxes :", pred_boxes.shape)

                    # Filter predictions
                    thresh = 0.4 
                    keep = scores > thresh
                    pred_boxes = pred_boxes[keep]
                    labels = labels[keep]
                    scores = scores[keep]

                    print("pred_boxes :", pred_boxes.shape)
                    print("labels :", labels)
                    print("scores :", scores)
                    # CHECK IMAGE INPUT ??
                    from .utils.visualization import Segmentation
                    filename='./TEST_MASK2FORMER_PRED.png'
                    visualization = Segmentation(-1, (self.pixel_mean.cpu(), self.pixel_std.cpu()), bgr_to_rgb=False)
                    # # mask
                    # visualization(images[index_vis].unsqueeze(0).cpu(), gt_msk.unsqueeze(0).cpu(), filename)
                    # box
                    visualization.plot_bbox(images[index_vis].unsqueeze(0).cpu(), pred_boxes.cpu(), filename)
                    self.i_debug+=1
                    if self.i_debug>3:
                        raise NotImplementedError("Visualize image[0]")
                    
                
            return processed_results

    def prepare_targets(self, targets, images, dataset_names=None):
        h_pad, w_pad = images.tensor.shape[-2:]

        new_targets = []
        for i, targets_per_image in enumerate(targets):
            
            if self.mask_captioning:
                gt_captions = targets_per_image.gt_captions
            else :
                gt_captions = None

            if dataset_names is not None:
                dataset = dataset_names[i]
            else:
                dataset = ""

            
            if self.use_masks :
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            if self.use_boxes:
                # no need to pad boxes
                gt_boxes = targets_per_image.gt_boxes.tensor
                # Normalize boxes
                gt_boxes[:, [0, 2]] /= w_pad
                gt_boxes[:, [1, 3]] /= h_pad
            
            obj={
                "labels": targets_per_image.gt_classes,
                "captions": gt_captions,
                "dataset": dataset,
                "masks": padded_masks if self.use_masks else None,
                "boxes": gt_boxes if self.use_boxes else None,
            }
            if not self.use_masks:
                obj.pop("masks")
            if not self.use_boxes:
                obj.pop("boxes")
            
            new_targets.append(obj)
            
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_object_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        scores = mask_cls[:,:-1].sigmoid()
        object_scores = F.softmax(mask_object_cls[0], dim=-1)[:, :-1]
        
        #####
        scores = (scores * object_scores) ** 0.5
        #####
        

        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        #topk_indices = topk_indices // self.sem_seg_head.num_classes
        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='trunc')
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image

        result.pred_classes = labels_per_image
        return result
    

    def box_instance_inference(self, mask_cls, mask_object_cls, box_pred, image_size):
        scores = mask_cls[:,:-1].sigmoid()
       
        object_scores = F.softmax(mask_object_cls[0], dim=-1)[:, :-1]
        
        ######
        scores = (scores * object_scores) ** 0.5
        ######

        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='trunc')
        box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = BitMasks(torch.zeros(box_pred.size(0), image_size[0], image_size[1]))
        result.pred_boxes = Boxes(box_pred)

        if self.box_xyxy :
            # Convert boxes to xyxy
            from utils.box_ops import box_xywh_to_xyxy
            result.pred_boxes.tensor = box_xywh_to_xyxy(result.pred_boxes.tensor)

        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        # mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        # result.scores = scores_per_image * mask_scores_per_image
        
        result.scores = scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_dvoc_inference(self, mask_cls, mask_object_cls, box_pred, image_size):
        scores, labels = mask_cls[:,:-1].sigmoid().max(dim=-1)
        
        object_scores = F.softmax(mask_object_cls[0], dim=-1)[:,0]
        
        ######
        scores = (scores * object_scores) ** 0.5
        ######


        scores_per_image, topk_indices = scores.topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        #topk_indices = topk_indices // self.sem_seg_head.num_classes
        box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = BitMasks(torch.zeros(box_pred.size(0), image_size[0], image_size[1]))
        result.pred_boxes = Boxes(box_pred)

        if self.box_xyxy :
            # Convert boxes to xyxy
            from utils.box_ops import box_xywh_to_xyxy
            result.pred_boxes.tensor = box_xywh_to_xyxy(result.pred_boxes.tensor)

        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        # mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        # result.scores = scores_per_image * mask_scores_per_image
        
        result.scores = scores_per_image
        result.pred_classes = labels_per_image
        return result