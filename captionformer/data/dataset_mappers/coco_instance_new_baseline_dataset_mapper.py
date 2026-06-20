# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/ovformer/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, Boxes, BoxMode

from pycocotools import mask as coco_mask

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train, no_crop=False):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    
    if no_crop:
        print("\n\n no crop for tuning \n\n")
        max_scale = 1.0
    
    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        no_crop=False,
        box_only=False,
        use_boxes=False,
        use_masks=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.no_crop = no_crop
        self.box_only = box_only
        self.use_boxes = use_boxes
        self.use_masks = use_masks
    
    @classmethod
    def from_config(cls, cfg, is_train=True, no_crop=False):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train, no_crop=no_crop)
        box_only = (cfg.MODEL.MASK_FORMER.BOX_MODE_ON == True and "vidstg" in cfg.DATASETS.TRAIN[0])
        use_boxes = cfg.MODEL.MASK_FORMER.USE_BOXES
        use_masks = cfg.MODEL.MASK_FORMER.USE_MASKS
        ret = {
            "is_train": is_train,
            "no_crop": no_crop,
            "box_only": box_only,
            "use_boxes": use_boxes,
            "use_masks": use_masks,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        try :
            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            # print("file_name : ", dataset_dict["file_name"])
            try :
                utils.check_image_size(dataset_dict, image)
            except Exception as e:
                print("image : ", dataset_dict["file_name"])
                print("image shape : ", image.shape)
                print("full dataset_dict : ", dataset_dict)
                raise e

            # TODO: get padding mask
            # by feeding a "segmentation mask" to the same transforms
            padding_mask = np.ones(image.shape[:2])

            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            # the crop transformation has default padding value 0 for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)

            image_shape = image.shape[:2]  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
            
            
            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    # Let's always keep mask
                    # if not self.mask_on:
                    #     anno.pop("segmentation", None)
                    if self.box_only:
                        anno.pop("segmentation", None)
                    anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                # print("annotations before filtering iscrowd : ", dataset_dict["annotations"])
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]

                # NOTE: does not support BitMask due to augmentation
                # Current BitMask cannot handle empty objects
                instances = utils.annotations_to_instances(annos, image_shape)

                # Need to filter empty instances first (due to augmentation)
                instances = utils.filter_empty_instances(instances)

                if (not self.box_only) and (not self.no_crop) and not(self.use_boxes):
                    # After transforms such as cropping are applied, the bounding box may no longer
                    # tightly bound the object. As an example, imagine a triangle object
                    # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                    # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                    # the intersection of original bounding box and the cropping box.
                    gt_boxes_xyxy = instances.gt_masks.get_bounding_boxes().tensor
                else :
                    gt_boxes_xyxy = instances.gt_boxes.tensor
                #Convert XYXY_ABS to XYWH_ABS (for image datasets)
                instances.gt_boxes = Boxes(BoxMode.convert(gt_boxes_xyxy, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS))
                

                # Generate masks from polygon
                h, w = instances.image_size
                if hasattr(instances, 'gt_masks'):
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                    instances.gt_masks = gt_masks
                elif self.use_masks and dataset_dict["dataset"] == "visualgenome":
                    num_instances = len(instances)
                    # Add full 1 gt mask if not present
                    instances.gt_masks = torch.ones((num_instances, image_shape[0], image_shape[1]), dtype=torch.uint8)
                dataset_dict["instances"] = instances
        except Exception as e:
            print("Error in COCOInstanceNewBaselineDatasetMapper : ", e)
            print("dataset_dict : ", dataset_dict)
            raise Exception("Error in COCOInstanceNewBaselineDatasetMapper") from e
        return dataset_dict
