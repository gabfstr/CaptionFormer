import os
import io
import logging
import contextlib

from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)

def load_bensmot_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
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
        
        # if "file_names" in vid_dict:
        record["file_names"] = [os.path.join(image_root, img_filename) for img_filename in vid_dict["file_names"]]
        assert len(record["file_names"]) == video_len , "len(record['file_names']) : {} != video_len : {}".format(len(record["file_names"]), video_len)
        sample_frames = list(range(1, video_len+1))


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

                obj["caption"] = anno.get("caption", None)


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


def register_bensmot_instances(name, metadata, json_file, image_root):
    """
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    DatasetCatalog.register(name, lambda: load_bensmot_json(
        json_file, image_root, name, extra_annotation_keys=['instance_id'], 
    ))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="bensmot", **metadata
    )

categories = [
    {'id': 1, 'name': 'person'},
]

def _get_bensmot_instances_meta():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}