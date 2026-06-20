# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

from .ovis import (
    register_ovis_instances,
    _get_ovis_instances_meta,
)

from .lvvis import (
    register_lvvis_instances,
    _get_lvvis_instances_meta,
)

from .vidstg import (
    register_vidstg_instances,
    _get_vidstg_instances_meta,
)

from .burst import (
    register_burst_instances,
    _get_burst_instances_meta,
)

from .bensmot import (
    register_bensmot_instances,
    _get_bensmot_instances_meta,
)

from .vln import (
    register_vln_instances,
    _get_vln_instances_meta,
)

from .smit import (
    register_smit_instances,
    _get_smit_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                   "ovis/annotations/train.json"),
    "ovis_val": ("ovis/valid",
                 "ovis/annotations/valid.json"),
    "ovis_test": ("ovis/test",
                  "ovis/annotations/test.json"),
}

# ==== Predefined splits for LVVIS ===========
_PREDEFINED_SPLITS_LVVIS = {
    "lvvis_train": ("LVVIS/train/JPEGImages",
                    "LVVIS/train/train_instances_nonovel.json"),
    "lvvis_val": ("LVVIS/val/JPEGImages",
                  "LVVIS/val/val_instances_.json"),
    "lvvis_test": ("LVVIS/test/JPEGImages",
                   "LVVIS/test/test_instances.json"),
    "lvviscap_train": ("LVVIS/train/JPEGImages",
                    "LVVIS/lvviscap_train_instances.json"),
    "lvviscap_val": ("LVVIS/val/JPEGImages",
                    "LVVIS/lvviscap_val_instances.json"),
    "lvviscap_test": ("LVVIS/test/JPEGImages",
                    "LVVIS/lvviscap_test_instances.json"),
    # Dev-only: 4-video subset for filter equivalence testing. Not for release.
    "lvviscap_val_4vids": ("LVVIS/val/JPEGImages",
                    "LVVIS/lvviscap_val_4vids.json"),
}


# ==== Predefined splits for VidSTG ===========
_PREDEFINED_SPLITS_VIDSTG = {
    "vidstg_train": ("VidSTG/video",
                    "VidSTG/annotations/train_instances_.json"),
    "vidstg_val": ("VidSTG/video",
                     "VidSTG/annotations/val_instances_.json"),
    "vidstg_train_200_frames": ("VidSTG/video",
                    "VidSTG/annotations/train_instances_200_frames_.json"),
    "vidstg_max200f_val": ("VidSTG/video",
                    "VidSTG/annotations/vidstg_max200f_val_instances_.json"),
    "vidstg_max200f_val_4v": ("VidSTG/video",
                    "VidSTG/annotations/vidstg_max200f_val_4v_instances_.json"),
}


# ==== Predefined splits for BURST ===========

_PREDEFINED_SPLITS_BURST= {
    "burst_val": ("burst/val",
                  "burst/b2y_val.json"),
}


# ==== Predefined splits for BENSMOT ===========

_PREDEFINED_SPLITS_BENSMOT = {
    "bensmot_train": ("bensmot/train", 
        "bensmot/annotations/train_instances_.json"),
    "bensmot_val": ("bensmot/test", 
        "bensmot/annotations/test_instances_.json"),
}

_PREDEFINED_SPLITS_VLN = {
    "vln_dvoc_train": ("VLN/uvo_videos_sparse_frames",
                            "VLN/annotations/vng_uvo_sparse_train_instances_.json"),
    "vln_dvoc_val": ("VLN/uvo_videos_sparse_frames",
                            "VLN/annotations/vng_uvo_sparse_val_instances_.json"),
    "vln_dvoc_val_extended": ("VLN/uvo_videos_sparse_frames",
                            "VLN/annotations/vng_uvo_sparse_val_extended_instances_.json"),
    # Dev-only: 4-video subset for filter equivalence testing. Not for release.
    "vln_dvoc_val_extended_4vids": ("VLN/uvo_videos_sparse_frames",
                            "VLN/annotations/vng_uvo_sparse_val_extended_4vids.json"),
}


_PREDEFINED_SPLITS_SMIT = {
    "smit_train": ("S-MiT/videos",
                   "S-MiT/annotations/train_instances_.json"),
}



def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ovis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_lvvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_lvvis_instances(
            key,
            _get_lvvis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_vidstg(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIDSTG.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vidstg_instances(
            key,
            _get_vidstg_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_burst(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BURST.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_burst_instances(
            key,
            _get_burst_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_bensmot(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BENSMOT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_bensmot_instances(
            key,
            _get_bensmot_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_vln(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VLN.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vln_instances(
            key,
            _get_vln_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_smit(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SMIT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_smit_instances(
            key,
            _get_smit_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)
    register_all_lvvis(_root)
    register_all_vidstg(_root)
    register_all_burst(_root)
    register_all_bensmot(_root)
    register_all_vln(_root)
    register_all_smit(_root)
