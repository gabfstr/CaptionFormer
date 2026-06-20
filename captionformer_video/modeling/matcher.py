# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer_video/modeling/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np

from detectron2.projects.point_rend.point_features import point_sample

from utils.box_ops import generalized_box_iou, l1_cost_matrix, box_xywh_to_xyxy, box_clamp_xyxy


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class VideoHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_object: float = 1, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, 
                 cost_bbox: float = 1, cost_giou: float = 1, by:str = 'mask', num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            cost_bbox: This is the relative weight of the L1 loss of the bounding box in the matching cost
            cost_giou: This is the relative weight of the GIoU loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_object = cost_object
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
        assert by in ['mask', 'box'], "by must be either 'mask' or 'box'"
        self.by_mask = by == 'mask'

        assert cost_object != 0 or cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    def linear_sum_assignment_with_inf(self, cost_matrix):
        cost_matrix = np.asarray(cost_matrix)
        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")

        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            min_values = values.min()
            max_values = values.max()
            m = min(cost_matrix.shape)

            positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
            if max_inf:
                place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
            elif min_inf:
                place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

            cost_matrix[np.isinf(cost_matrix)] = place_holder
        return linear_sum_assignment(cost_matrix)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_class_prob = outputs["pred_logits"][b][:, :-1].sigmoid()  # [num_queries, num_classes]
            out_object_prob = outputs["pred_object_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_prob = torch.cat([(out_class_prob * out_object_prob[:, 0:1]) ** 0.5, out_object_prob[:, 1:2]], dim=1)

            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]

            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(self.linear_sum_assignment_with_inf(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
    
    @torch.no_grad()
    def box_memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_class_prob = outputs["pred_logits"][b][:, :-1].sigmoid()  # [num_queries, num_classes]
            out_object_prob = outputs["pred_object_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_prob = torch.cat([(out_class_prob * out_object_prob[:, 0:1]) ** 0.5, out_object_prob[:, 1:2]], dim=1)

            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_boxes = outputs["pred_boxes"][b].transpose(0,1)  # [T, num_queries, 4]
            # gt masks are already padded when preparing target
            tgt_boxes = targets[b]["boxes"].to(out_boxes).transpose(0,1)  # [T, num_gts, 4]
            
            num_frames, num_gts = tgt_boxes.shape[:2]

            
            cost_giou = torch.zeros((num_queries, num_gts), device=out_boxes.device)
            cost_bbox = torch.zeros((num_queries, num_gts), device=out_boxes.device)
            
            for i in range(num_frames):
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_boxes[i], tgt_boxes[i], p=1)

                # Compute the giou cost betwen boxes
                cost_giou = 1 - generalized_box_iou(
                    box_clamp_xyxy(box_xywh_to_xyxy(out_boxes[i])),
                    box_xywh_to_xyxy(tgt_boxes[i])
                )

            cost_giou /= num_frames
            cost_bbox /= num_frames

            # Final cost matrix
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(self.linear_sum_assignment_with_inf(C))
            

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if self.by_mask:
            return self.memory_efficient_forward(outputs, targets)
        else:
            return self.box_memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_object: {}".format(self.cost_object),
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)



# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np

from detectron2.projects.point_rend.point_features import point_sample

from utils.box_ops import generalized_box_iou, l1_cost_matrix, box_xywh_to_xyxy, box_clamp_xyxy


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule








class VideoHungarianTemporalMatcher(nn.Module):
    """This class computes an assignment between the temporal predictions of the network (time t, time t+1)

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_embed: float = 1, cost_object: float = 1, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, 
                 cost_bbox: float = 1, cost_giou: float = 1, by:str = 'mask', num_points: int = 0):
        """Creates the matcher

        Params:
            cost_embed: This is the relative weight of the similarity error between embeddings in the matching cost
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            cost_bbox: This is the relative weight of the L1 loss of the bounding box in the matching cost
            cost_giou: This is the relative weight of the GIoU loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_embed = cost_embed
        self.cost_object = cost_object
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
        assert by in ['mask', 'box'], "by must be either 'mask' or 'box'"
        self.by_mask = (by == 'mask')

        assert cost_embed!=0 or cost_object != 0 or cost_class != 0 or cost_mask != 0 or cost_dice != 0 or cost_bbox!=0 or cost_giou!=0, "all costs cant be 0"

        self.num_points = num_points

    def linear_sum_assignment_with_inf(self, cost_matrix):
        cost_matrix = np.asarray(cost_matrix)
        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")

        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            min_values = values.min()
            max_values = values.max()
            m = min(cost_matrix.shape)

            positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
            if max_inf:
                place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
            elif min_inf:
                place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

            cost_matrix[np.isinf(cost_matrix)] = place_holder
        return linear_sum_assignment(cost_matrix)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_class_prob = outputs["pred_logits"][b][:, :-1].sigmoid()  # [num_queries, num_classes]
            out_object_prob = outputs["pred_object_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_prob = torch.cat([(out_class_prob * out_object_prob[:, 0:1]) ** 0.5, out_object_prob[:, 1:2]], dim=1)

            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]


            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(self.linear_sum_assignment_with_inf(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
    
    @torch.no_grad()
    def box_memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        num_queries = outputs["pred_logits"].shape[0]
        
            
        out_class_prob = outputs["pred_logits"][:, :-1].sigmoid()  # [num_queries, num_classes]
        out_object_prob = outputs["pred_object_logits"].softmax(-1)  # [num_queries, num_classes]
        out_prob = torch.cat([(out_class_prob * out_object_prob[:, 0:1]) ** 0.5, out_object_prob[:, 1:2]], dim=1)

        tgt_class_prob = targets["pred_logits"][:, :-1].sigmoid()  # [num_queries, num_classes]
        tgt_object_prob = targets["pred_object_logits"].softmax(-1)  # [num_queries, num_classes]
        tgt_prob = torch.cat([(tgt_class_prob * tgt_object_prob[:, 0:1]) ** 0.5, tgt_object_prob[:, 1:2]], dim=1)
        
        # Compute similarity between class predictions
        cost_class = torch.cdist(out_prob, tgt_prob, p=1)
        # This is the l1 distance between the embeddings classification scores (combined with object scores)


        # # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # # but approximate it in 1 - proba[target class].
        # # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class = -out_prob[:, tgt_ids]

        # Compute similarity between embeddings
        out_embds = outputs["pred_embds"]  # [num_queries, embed_dim]            
        out_embds = out_embds / out_embds.norm(dim=1)[:, None]
        tgt_embds = targets["pred_embds"]  # [num_queries, embed_dim]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(out_embds, tgt_embds.transpose(0,1))  # (num_queries,num_queries)

        cost_embd = 1 - cos_sim

        out_boxes = outputs["pred_boxes"].transpose(0,1)  # [T, num_queries, 4]
        tgt_boxes = targets["pred_boxes"].transpose(0,1)  # [T, num_queries, 4]
        
        num_frames, num_gts = tgt_boxes.shape[:2]
        
        cost_giou = torch.zeros((num_queries, num_gts), device=out_boxes.device)
        cost_bbox = torch.zeros((num_queries, num_gts), device=out_boxes.device)

        # Compute the cost between bbox in the last frame of src vs 1st frame of tgt
        last_out_boxes = out_boxes[-1]
        first_tgt_boxes = tgt_boxes[0]
        
        # L1 cost
        cost_bbox = torch.cdist(last_out_boxes, first_tgt_boxes, p=1)

        # GIoU cost
        cost_giou = 1 - generalized_box_iou(
            box_clamp_xyxy(box_xywh_to_xyxy(last_out_boxes)),
            box_clamp_xyxy(box_xywh_to_xyxy(first_tgt_boxes))
        )

        
        # Final cost matrix
        C = (
            + self.cost_class * cost_class
            + self.cost_embed * cost_embd
            + self.cost_giou * cost_giou
            + self.cost_bbox * cost_bbox
        )
        C = C.reshape(num_queries, -1).cpu()
        indices = self.linear_sum_assignment_with_inf(C)[1]
            
        return indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if self.by_mask:
            return self.memory_efficient_forward(outputs, targets)
        else:
            return self.box_memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_embed: {}".format(self.cost_embed),
            "cost_object: {}".format(self.cost_object),
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
