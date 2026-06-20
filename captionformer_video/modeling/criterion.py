# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer_video/modeling/criterion.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from captionformer.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from captionformer.modeling.util import load_class_freq, get_fed_loss_inds


from fvcore.nn import smooth_l1_loss
import utils.box_ops as box_ops


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


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        empty_weight[-1] = 0
        empty_object_weight = torch.ones(2)
        empty_object_weight[-1] = 0.4
        self.register_buffer("empty_weight", empty_weight)
        self.register_buffer("empty_object_weight", empty_object_weight)
        if num_classes == 80:
            # Vidstg dataset
            freq_weight = load_class_freq('datasets/metadata/vidstg_train_cat_info.json', 0.5)
        elif num_classes == 1203:
            # LVIS dataset
            freq_weight = load_class_freq('datasets/metadata/lvis_v1_train_cat_info.json', 0.5)
        elif num_classes == 1:
            # Bensmot dataset
            freq_weight = load_class_freq('datasets/metadata/bensmot_train_cat_info.json', 0.5)
            freq_weight = None
        else :
            # LVVIS dataset by default
            freq_weight = load_class_freq('datasets/metadata/lvvis_train_cat_info.json', 0.5)
        
        if freq_weight is not None :
            self.register_buffer('freq_weight', freq_weight)
        else :
            self.freq_weight = None

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        B, Q = src_logits.shape[0], src_logits.shape[1]
        src_object_logits = outputs["pred_object_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target_object_classes = (target_classes == self.num_classes).long()
        target_classes_binary = F.one_hot(target_classes, num_classes=self.num_classes + 1)
        target_classes_binary = target_classes_binary[:, :, :self.num_classes].float()
        
        if self.freq_weight is not None :
            appeared = get_fed_loss_inds(
                target_classes,
                num_sample_cats=50,
                C=self.num_classes,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(self.num_classes + 1)
            appeared_mask[appeared] = 1  # C + 1
            appeared_mask = appeared_mask[:self.num_classes]

            weight = 1
            fed_w = appeared_mask.view(1, 1, self.num_classes).expand(B, Q, self.num_classes)
            weight = weight * fed_w.float()

        loss_ce = F.binary_cross_entropy_with_logits(src_logits[:, :, :-1], target_classes_binary,
                                                     reduction='none')  # B x C

        if self.freq_weight is not None :
            loss_ce = 1.7 * torch.sum(loss_ce * weight * (1 - target_object_classes[:, :, None])) / (B * Q)
        else :
            loss_ce = 1.7 * torch.sum(loss_ce * (1 - target_object_classes[:, :, None])) / (B * Q)

        loss_object_ce = F.cross_entropy(src_object_logits.transpose(1, 2), target_object_classes,
                                         self.empty_object_weight)

        losses = {"loss_ce": loss_ce, "loss_object_ce": loss_object_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
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

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bboxes: the smooth l1 loss and the giou loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_boxes" in outputs

        src_idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"].float()
        src_boxes = src_boxes[src_idx]

        # Modified to handle video
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)]).to(src_boxes)

        # TODO use valid to mask invalid areas due to padding in loss
        # target_boxes, valid = nested_tensor_from_tensor_list(boxes).decompose()
        # target_boxes = target_boxes.to(src_boxes)
    
        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 4
        src_boxes = src_boxes.flatten(0, 1)
        target_boxes = target_boxes.flatten(0, 1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_l1_box'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_clamp_xyxy(box_ops.box_xywh_to_xyxy(src_boxes)), # clamp to not have invalid x2,y2 value (x2<x1 or y2<y1) 
            box_ops.box_xywh_to_xyxy(target_boxes)))
        losses['loss_giou_box'] = loss_giou.sum() / num_boxes

    
        del src_boxes
        del target_boxes
        return losses



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
    
    def _permute_tgt_boxes(self, tgt_boxes, tgt_idx:tuple):
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

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
