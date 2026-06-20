
import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from .vidstg_categories import VIDSTG_CATEGORIES
"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_vidstg_json", "register_vidstg_instances"]


def _get_vidstg_instances_meta():
    thing_ids = [k["id"] for k in VIDSTG_CATEGORIES]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VIDSTG_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret



def load_vidstg_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from .ytvis_api.lvvis import LVVIS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = LVVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))


    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        
    
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)} 
        meta.thing_dataset_id_to_contiguous_id = id_map 
    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}

        # Get len and fps 
        video_len = vid_dict["length"]
       
        if "file_names" in vid_dict:
            record["file_names"] = [os.path.join(image_root, img_filename) for img_filename in vid_dict["file_names"]]
            assert len(record["file_names"]) == video_len , "len(record['file_names']) : {} != video_len : {}".format(len(record["file_names"]), video_len)
            sample_frames = list(range(1, video_len+1))
        else : 
            video_file_name = vid_dict["file_name"].split(".mp4")[0]
            # fps = vid_dict["fps"]
            # n_seconds = video_len / fps
            
            # if dataset_name == "vidstg_train":
            #     target_number_of_frames = int(5 * n_seconds)  # Desired number of frames at 5 FPS
                
            #     # Compute the sampling interval
            #     sampling_interval = video_len / target_number_of_frames
            #     # Sample frames to achieve 5 FPS
            #     sample_frames = [round(1 + i * sampling_interval) for i in range(target_number_of_frames)]
            # else : 
            #     sample_frames = list(range(1, video_len+1)) 
            
            sample_frames = list(range(1, video_len+1))   
            record["file_names"] = [os.path.join(image_root, video_file_name , f"{frame_idx:04d}.jpg") for frame_idx in sample_frames]
            
            feature_folder = os.path.join(image_root.replace("video", "features"))
            # Features are stored in batches of 64 frames : feat_filename_1of1.pth
            batch_size = 64
            batch_idx = (np.array(sample_frames) - 1) // batch_size
            record["feature_path"] = [
                os.path.join(feature_folder, video_file_name, 
                                f"feat_{video_file_name.split('/')[-1]}_{idx+1}of{1+len(batch_idx)//batch_size}.pth") 
                                for idx in batch_idx]
        

        # record["file_names"] = [os.path.join(image_root, '/'.join(vid_dict["file_names"][i].split('\\')[-2:])) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = len(sample_frames) # new length
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []

        for nb_sampled in sample_frames:
            frame_idx = nb_sampled-1
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}


                _box = anno.get("bbox", None)
                # if not ( _box and _box[frame_idx]):

                # Do not add object with no annotation to the frame record 
                if not (_box and _box[frame_idx]) or _box[frame_idx] is None:
                    # print("continue")
                    continue
                
                box = _box[frame_idx]
                obj["bbox"] = box
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                _area = anno.get("areas", None)
                area = _area[frame_idx]
                obj["area"] = area

                obj["caption"] = anno.get("caption", '')
                if isinstance(obj["caption"], list):
                    obj["caption"] = obj["caption"][frame_idx]
                if obj["caption"] is None:
                    obj["caption"] = ''

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                
                frame_objs.append(obj)


            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_vidstg_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_vidstg_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )

