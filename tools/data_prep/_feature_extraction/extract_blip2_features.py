# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""Extract BLIP-2 (EVA-ViT-g) visual features for video datasets — multi-GPU.

One unified entry point for LVVIS, VidSTG, VLN, BenSMOT. Each dataset gets a
small "profile" capturing its (annotation_path, image_root, features_root,
image-paths-from-vid-dict logic). The encode loop, distributed plumbing, and
index-mapping write/merge are shared and copied verbatim from the original
per-dataset scripts.

For LVIS (image-based, different API and output layout), see
`tools/data_prep/_feature_extraction/lviscap_extract_blip2_features.py`.

Usage (launch from repo root):
    python tools/data_prep/_feature_extraction/extract_blip2_features.py \
        --dataset {lvvis,vidstg,vln,bensmot} --split {train,val,test}

The script uses `torch.distributed` (NCCL, localhost) over all GPUs it sees.

Output layout (per dataset):
    LVVIS:    datasets/LVVIS/<split>/features/<video_id>/feat_<vid>_<B>of<N>.pth
              datasets/LVVIS/<split>/features/index_mapping.json
    VidSTG:   datasets/VidSTG/features/<video_file_name>/feat_...
              datasets/VidSTG/features/index_mapping.json
    VLN:      datasets/VLN/features/<split>/<video_id>/feat_...
              datasets/VLN/features/<split>/index_mapping.json
    BenSMOT:  datasets/bensmot/features/<split>/<seq_name>/feat_...
              datasets/bensmot/features/<split>/index_mapping.json
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Make captionformer importable when launched from repo root
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(parent_dir)
from captionformer.modeling.captioning_head.eva_vit import create_eva_vit_g  # noqa: E402
from captionformer.data_video.datasets.ytvis_api.lvvis import LVVIS  # noqa: E402


BATCH_SIZE = 64
IMG_SIZE = 364


# ----------------------------------------------------------------------
# Per-dataset image-path / id / output-folder builders.
# Each is a verbatim copy of the logic from the original per-dataset script,
# so behaviour (final file paths on disk + index_mapping contents) matches.
# ----------------------------------------------------------------------

# ---- LVVIS -----------------------------------------------------------------
def lvvis_image_paths(vid_dict, image_root):
    return [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]


def lvvis_video_id_name(vid_dict):
    return os.path.dirname(vid_dict["file_names"][0])


def lvvis_features_folder(image_root, video_id_name, split):
    return os.path.join(image_root.replace("JPEGImages/", "features/"), video_id_name)


# ---- VidSTG ----------------------------------------------------------------
def vidstg_image_paths(vid_dict, image_root):
    video_file_name = vid_dict["file_name"].split(".mp4")[0]
    video_len = vid_dict["length"]
    sample_frames = list(range(1, video_len + 1))
    return [os.path.join(image_root, video_file_name, f"{frame_idx:04d}.jpg") for frame_idx in sample_frames]


def vidstg_video_id_name(vid_dict):
    # Original used video_file_name as the folder key (full sub-path), and
    # video_id_name (basename) as the .pth filename stem — preserve both.
    return vid_dict["file_name"].split(".mp4")[0]


def vidstg_features_folder(image_root, video_id_name, split):
    return os.path.join(image_root.replace("video", "features"), video_id_name)


def vidstg_pth_stem(vid_dict):
    # `feat_<basename>_<B>of<N>.pth` (basename, NOT the full sub-path key)
    return os.path.basename(vid_dict["file_name"]).split(".mp4")[0]


# ---- VLN -------------------------------------------------------------------
def vln_image_paths(vid_dict, image_root):
    return [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]


def vln_video_id_name(vid_dict):
    return os.path.dirname(vid_dict["file_names"][0].split("/imgs/")[0])


def vln_features_folder(image_root, video_id_name, split):
    return os.path.join(image_root.replace("uvo_videos_sparse_frames", f"features/{split}"), video_id_name)


# ---- BenSMOT ---------------------------------------------------------------
def bensmot_image_paths(vid_dict, image_root):
    return [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]


def bensmot_video_id_name(vid_dict):
    return vid_dict["file_names"][0].split("/imgs/")[0].split("/")[-1]


def bensmot_features_folder(image_root, video_id_name, split):
    return os.path.join(image_root.replace(split, f"features/{split}"), video_id_name)


# ----------------------------------------------------------------------
# DATASET_PROFILES — one entry per supported dataset.
# `paths[split]` is a dict of paths used by the loop. For datasets where the
# original script only ever ran on one split (e.g. vidstg train), only that
# split's paths are populated here — passing `--split` for a missing split
# raises an explicit error.
# ----------------------------------------------------------------------
DATASET_PROFILES = {
    'lvvis': {
        'paths': {
            'train': {
                'ann':                  './datasets/LVVIS/train_instances_.json',
                'image_root':           './datasets/LVVIS/train/JPEGImages/',
                'index_mapping_file':   './datasets/LVVIS/train/features/index_mapping.json',
                'rank_mapping_template':'./datasets/LVVIS/train/features/index_mapping_rank{rank}.json',
                'error_log':            './datasets/LVVIS/train/features/feature_extraction_errors.txt',
            },
        },
        'image_paths_fn':     lvvis_image_paths,
        'video_id_name_fn':   lvvis_video_id_name,
        'features_folder_fn': lvvis_features_folder,
        'reverse_vids': False,
    },
    'vidstg': {
        'paths': {
            'train': {
                'ann':                   './datasets/VidSTG/annotations/train_instances_.json',
                'image_root':            './datasets/VidSTG/video/',
                'index_mapping_file':    './datasets/VidSTG/features/index_mapping.json',
                'rank_mapping_template': './datasets/VidSTG/features/index_mapping_rank{rank}.json',
                'error_log':             './datasets/VidSTG/features/feature_extraction_errors.txt',
            },
        },
        'image_paths_fn':     vidstg_image_paths,
        'video_id_name_fn':   vidstg_video_id_name,
        'features_folder_fn': vidstg_features_folder,
        'pth_stem_fn':        vidstg_pth_stem,           # VidSTG-only: .pth uses basename, folder uses sub-path
        'reverse_vids': False,
    },
    'vln': {
        'paths': {
            split: {
                'ann':                   f'./datasets/VLN/annotations/vng_uvo_sparse_{split}_instances_.json',
                'image_root':            './datasets/VLN/uvo_videos_sparse_frames/',
                'index_mapping_file':    f'./datasets/VLN/features/{split}/index_mapping.json',
                'rank_mapping_template': f'./datasets/VLN/features/{split}/index_mapping_rank{{rank}}.json',
                'error_log':             f'./datasets/VLN/features/{split}/feature_extraction_errors.txt',
            } for split in ('train', 'val')
        },
        'image_paths_fn':     vln_image_paths,
        'video_id_name_fn':   vln_video_id_name,
        'features_folder_fn': vln_features_folder,
        'reverse_vids': False,
    },
    'bensmot': {
        'paths': {
            split: {
                'ann':                   f'./datasets/bensmot/annotations/{split}_instances_.json',
                'image_root':            f'./datasets/bensmot/{split}/',
                'index_mapping_file':    './datasets/bensmot/features/index_mapping.json',
                'rank_mapping_template': './datasets/bensmot/features/index_mapping_rank{rank}.json',
                'error_log':             './datasets/bensmot/features/feature_extraction_errors.txt',
            } for split in ('train', 'test')
        },
        'image_paths_fn':     bensmot_image_paths,
        'video_id_name_fn':   bensmot_video_id_name,
        'features_folder_fn': bensmot_features_folder,
        'reverse_vids': True,    # original bensmot script reverses the vids list before per-rank slicing
    },
}


# ----------------------------------------------------------------------
# Shared infrastructure (copied verbatim from the original per-dataset scripts)
# ----------------------------------------------------------------------
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def merge_index_mappings(world_size, rank_mapping_template, final_output_file):
    index_mapping = {}
    for rank in range(world_size):
        rank_file = rank_mapping_template.format(rank=rank)
        print("Merging index_mapping with rank file ", rank_file)
        if os.path.exists(rank_file):
            with open(rank_file, "r") as f:
                index_mapping.update(json.load(f))
    with open(final_output_file, "w") as f:
        json.dump(index_mapping, f)
    print(f"Final merged index saved to {final_output_file}")


def build_visual_encoder(rank):
    visual_encoder = create_eva_vit_g(
        IMG_SIZE, drop_path_rate=0, use_checkpoint=False, precision="fp32",
    ).to(rank)
    visual_encoder = nn.parallel.DistributedDataParallel(visual_encoder, device_ids=[rank])
    visual_encoder.eval()

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return visual_encoder, transform


# ----------------------------------------------------------------------
# Shared loop body — equivalent across all 4 video datasets in the originals.
# ----------------------------------------------------------------------
def main(rank, world_size, dataset, split):
    setup_distributed(rank, world_size)

    profile = DATASET_PROFILES[dataset]
    if split not in profile['paths']:
        raise ValueError(f"Dataset '{dataset}' has no '{split}' split configured "
                         f"(available: {list(profile['paths'].keys())}).")
    cfg = profile['paths'][split]

    ann_path = cfg['ann']
    image_root = cfg['image_root']
    error_log_file = cfg['error_log']
    index_mapping_file = cfg['index_mapping_file']
    rank_mapping_template = cfg['rank_mapping_template']

    ytvis_api = LVVIS(ann_path)
    vid_ids = sorted(ytvis_api.vids.keys())
    vids = ytvis_api.loadVids(vid_ids)

    if rank == 0:
        print(f"[{dataset}/{split}] loaded {len(vids)} videos from {ann_path}")
        print("Initializing visual encoder...")

    visual_encoder, transform = build_visual_encoder(rank)

    try:
        with open(error_log_file, "r") as f:
            error_files = set(line.strip() for line in f)
    except FileNotFoundError:
        error_files = set()
    try:
        with open(index_mapping_file, "r") as f:
            index_mapping = json.load(f)
    except FileNotFoundError:
        index_mapping = {}

    new_errors = set()

    # bensmot original reverses vids before per-rank slicing — preserve.
    if profile.get('reverse_vids', False):
        vids = vids[::-1]
    vids = vids[rank::world_size]

    image_paths_fn = profile['image_paths_fn']
    video_id_name_fn = profile['video_id_name_fn']
    features_folder_fn = profile['features_folder_fn']
    pth_stem_fn = profile.get('pth_stem_fn', None)   # only vidstg has one

    with tqdm(total=len(vids), desc=f"Rank {rank} Processing Videos") as pbar:
        for vid_dict in vids:
            image_paths = image_paths_fn(vid_dict, image_root)
            video_id_name = video_id_name_fn(vid_dict)
            new_file_folder = features_folder_fn(image_root, video_id_name, split)
            pth_stem = pth_stem_fn(vid_dict) if pth_stem_fn is not None else video_id_name

            os.makedirs(new_file_folder, exist_ok=True)
            num_batches = math.ceil(len(image_paths) / BATCH_SIZE)
            updated_index = False
            vid_index_mapping = index_mapping.get(video_id_name, {})

            for batch_index in range(1, num_batches + 1):
                output_filename = os.path.join(
                    new_file_folder, f"feat_{pth_stem}_{batch_index}of{num_batches}.pth")

                batch_features = []
                start_index = (batch_index - 1) * BATCH_SIZE
                end_index = min(batch_index * BATCH_SIZE, len(image_paths))
                for i in range(start_index, end_index):
                    img_file_name = image_paths[i]
                    current_index = i % BATCH_SIZE
                    vid_index_mapping[i] = {"file_name": output_filename, "index": current_index}
                    updated_index = True
                    if os.path.exists(output_filename):
                        if output_filename in error_files:
                            print("Processing images from error file:", output_filename)
                            error_files.remove(output_filename)
                            with open(error_log_file, "w") as f:
                                for line in error_files:
                                    f.write(line + "\n")
                        else:
                            continue
                    image = Image.open(img_file_name).convert("RGB")
                    image = transform(image).unsqueeze(0).to(rank)
                    with torch.no_grad():
                        features = visual_encoder(image).cpu()
                    batch_features.append(features)

                if batch_features:
                    try:
                        batch_features = torch.cat(batch_features, dim=0)
                        torch.save(batch_features, output_filename)
                        print(f"Rank {rank}: Saved features to {output_filename}")
                    except Exception as e:
                        print(f"Error processing {img_file_name} on rank {rank}: {e}")
                        new_errors.add(output_filename)
                        os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
                        with open(error_log_file, "a") as ef:
                            ef.write(f"{output_filename}\n")

            index_mapping[video_id_name] = vid_index_mapping
            if updated_index:
                if world_size > 1:
                    rank_file = rank_mapping_template.format(rank=rank)
                    os.makedirs(os.path.dirname(rank_file), exist_ok=True)
                    with open(rank_file, "w") as f:
                        json.dump(index_mapping, f)
                else:
                    os.makedirs(os.path.dirname(index_mapping_file), exist_ok=True)
                    with open(index_mapping_file, "w") as f:
                        json.dump(index_mapping, f)

            pbar.update(1)

    dist.barrier()

    if world_size <= 1:
        with open(index_mapping_file, "w") as f:
            json.dump(index_mapping, f)
    else:
        if rank == 0:
            print(f"Finished processing all videos. Errors files stored in {error_log_file}")
            print("now merging index mappings")
            merge_index_mappings(world_size, rank_mapping_template, index_mapping_file)

    cleanup_distributed()


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument('--dataset', required=True, choices=list(DATASET_PROFILES.keys()))
    ap.add_argument('--split', default='train',
                    help='Split to extract (defaults vary per dataset; see DATASET_PROFILES).')
    args = ap.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.dataset, args.split), nprocs=world_size, join=True)


if __name__ == "__main__":
    cli()
