# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/antoyang/TubeDETR/blob/main/util/box_ops.py
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import numpy as np
from torchvision.ops.boxes import box_area
from typing import Tuple

#### Bounding box utilities imported from torchvision and converted to numpy
def np_box_area(boxes: np.array) -> np.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = np_box_area(boxes1)
    area2 = np_box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def np_box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(boxes):
    x1, y1, w, h = boxes.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)

def box_xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)

def box_clamp_xyxy(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    b = [x1, y1, torch.max(x1, x2), torch.max(y1, y2)]
    return torch.stack(b, dim=-1)
    


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2, eps=1e-7):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=eps)
    return iou, union


def generalized_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area.clamp(min=eps)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)




def l1_cost_matrix(pred_boxes, gt_boxes):
    """
    Compute the L1 cost matrix between predicted and ground truth bounding boxes.

    Args:
        pred_boxes (torch.Tensor): Tensor of shape [N, 4] for predicted boxes.
        gt_boxes (torch.Tensor): Tensor of shape [M, 4] for ground truth boxes.

    Returns:
        torch.Tensor: Cost matrix of shape [N, M], where each entry (i, j) is
                      the L1 cost between pred_boxes[i] and gt_boxes[j].
    """
    # pred_boxes: [N, 4]
    # gt_boxes: [M, 4]

    # Reshape for broadcasting: [N, 1, 4] and [1, M, 4]
    pred_boxes = pred_boxes.unsqueeze(1)  # [N, 1, 4]
    gt_boxes = gt_boxes.unsqueeze(0)      # [1, M, 4]

    # Compute L1 loss (absolute difference)
    l1_cost = torch.abs(pred_boxes - gt_boxes).sum(dim=2)  # [N, M]

    return l1_cost



def attn_masking_from_bbox(bbox, attn_mask_target_size, num_heads=8):
    """
    Create an attention mask from bounding box coordinates.
    Input : bbox : [batch_size, num_queries, 4] : (x_min, y_min, width, height) between 0 and 1
            attn_mask_target_size : Tuple[int, int] : (height, width)
    Output : attn_mask : [batch_size * num_heads, num_queries, num_frames*target_size[0]*target_size[1]] : boolean mask
    """

    if bbox.ndim == 2:
        bbox = bbox.unsqueeze(1)

    # Scale box coordinates to attention mask size
    x_min = (bbox[..., 0] * attn_mask_target_size[1]).clamp(0, attn_mask_target_size[1] - 1).long()
    y_min = (bbox[..., 1] * attn_mask_target_size[0]).clamp(0, attn_mask_target_size[0] - 1).long()
    x_max = (x_min + bbox[..., 2] * attn_mask_target_size[1]).clamp(0, attn_mask_target_size[1] - 1).long()
    y_max = (y_min + bbox[..., 3] * attn_mask_target_size[0]).clamp(0, attn_mask_target_size[0] - 1).long()

    if bbox.ndim == 3:
        # Create a grid for mask regions
        batch_size, num_queries = bbox.shape[:2]
        attn_mask = torch.ones(
            (batch_size, num_queries, attn_mask_target_size[0], attn_mask_target_size[1]),
            device=bbox.device,
        )
        
        # Create boolean masks for each box region
        y_coords = torch.arange(attn_mask_target_size[0], device=bbox.device).view(1, 1, -1, 1)
        x_coords = torch.arange(attn_mask_target_size[1], device=bbox.device).view(1, 1, 1, -1)


    elif bbox.ndim == 4:
        batch_size, num_queries, num_frames = bbox.shape[:3]
        attn_mask = torch.ones(
            (batch_size, num_queries, num_frames, attn_mask_target_size[0], attn_mask_target_size[1]),
            device=bbox.device,
        )
        # Create boolean masks for each box region
        y_coords = torch.arange(attn_mask_target_size[0], device=bbox.device).view(1, 1, 1, -1, 1)
        x_coords = torch.arange(attn_mask_target_size[1], device=bbox.device).view(1, 1, 1, 1, -1)
    

    # Check if coordinates fall within the box regions
    in_y = (y_coords >= y_min.unsqueeze(-1).unsqueeze(-1)) & (y_coords < y_max.unsqueeze(-1).unsqueeze(-1))
    in_x = (x_coords >= x_min.unsqueeze(-1).unsqueeze(-1)) & (x_coords < x_max.unsqueeze(-1).unsqueeze(-1))

    # Combine masks for valid regions
    box_mask = in_y & in_x
    attn_mask = attn_mask * (~box_mask)  # Set valid regions to 0 (allowed)

    # Flatten and repeat for multi-head attention
    attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(1, num_heads, 1, 1).flatten(0, 1).bool()
    attn_mask = attn_mask.detach()

    return attn_mask