# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""Extract CLIP image features for VidSTG splits.

Iterates over the `videos[].file_names` field of a VidSTG annotation JSON,
encodes each frame with the CLIP ViT-B/32 visual encoder, saves a single
{frame_absolute_path: feature_tensor} pickle per split.

Train mode supports `--chunk_id` / `--num_chunks` for splitting the work
across multiple jobs (then merge with `--merge_chunks`). For val, just one
pass — much smaller, no chunking needed.

Run from the repo root. The resulting `.pkl` is loaded by the model via
`MODEL.MASK_FORMER.CLIP_IMAGE_PATH` during training/inference.
"""
import argparse
import json
import os

import torch
from clip import clip
from PIL import Image
from tqdm import tqdm


def encode_videos(videos, file_dir, model, preprocess, device, sample_all=True):
    """Encode CLIP features for every frame referenced by `file_names` in each video.
    With `sample_all=True`, iterate all frames `1..vid_len` (used for full-train mode).
    With `sample_all=False`, use the exact `file_names` list from the annotation
    (used for the `max200f` / val_200_frames pre-subsampled variants).
    """
    dic = {}
    for video in tqdm(videos):
        video_file_name = video["file_name"].split(".mp4")[0]
        if sample_all:
            vid_len = video["length"]
            sample_frames = list(range(1, vid_len + 1))
        else:
            sample_frames = [int(x.split("/")[-1].split(".jpg")[0]) for x in video["file_names"]]
        vid_file_names = [
            os.path.join(file_dir, video_file_name, f"{frame_idx:04d}.jpg")
            for frame_idx in sample_frames
        ]
        for path in vid_file_names:
            image_clip = preprocess(Image.open(path)).unsqueeze(0).to(device)
            feature_clip = model.encode_image(image_clip)
            dic[path] = feature_clip
    return dic


def save_split(json_path, file_dir, save_path, model, preprocess, device,
               sample_all=True, chunk_id=None, num_chunks=1):
    print(f"\n=== {os.path.basename(save_path)} ===")
    print(f"  ann:  {json_path}")
    print(f"  imgs: {file_dir}")
    data = json.load(open(json_path, "r"))
    videos = data["videos"]
    print(f"  total videos: {len(videos)}")

    if chunk_id is not None:
        chunk_size = len(videos) // num_chunks
        start = chunk_id * chunk_size
        end = (chunk_id + 1) * chunk_size if chunk_id < num_chunks - 1 else len(videos)
        videos = videos[start:end]
        save_path = save_path.replace(".pkl", f"_chunk{chunk_id+1}of{num_chunks}.pkl")
        print(f"  chunk {chunk_id+1}/{num_chunks}: videos[{start}:{end}] ({len(videos)} videos)")
    print(f"  out:  {save_path}")

    dic = encode_videos(videos, file_dir, model, preprocess, device, sample_all=sample_all)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dic, save_path)
    print(f"  saved {len(dic)} features → {save_path}")


def merge_chunks(out_path, num_chunks):
    """Merge chunk files {out_path}_chunkNofM.pkl into the single {out_path}."""
    chunk_paths = [out_path.replace(".pkl", f"_chunk{i+1}of{num_chunks}.pkl") for i in range(num_chunks)]
    missing = [p for p in chunk_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing chunk files: {missing}")
    print(f"merging {num_chunks} chunks → {out_path}")
    dic = {}
    for p in chunk_paths:
        dic.update(torch.load(p))
    torch.save(dic, out_path)
    print(f"  merged {len(dic)} features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", default="./datasets/VidSTG/annotations/train_instances_.json")
    ap.add_argument("--val_ann",   default="./datasets/VidSTG/annotations/val_instances_.json")
    ap.add_argument("--video_dir", default="./datasets/VidSTG/video/")
    ap.add_argument("--train_out", default="./datasets/metadata/vidstg_train_clip_feature.pkl")
    ap.add_argument("--val_out",   default="./datasets/metadata/vidstg_val_clip_feature.pkl")
    ap.add_argument("--model_path", default="ViT-B/32",
                    help="CLIP model name (auto-downloaded to ~/.cache/clip/) "
                         "or path to a .pt file.")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_val", action="store_true")
    # chunked train extraction (optional)
    ap.add_argument("--chunk_id", type=int, default=None,
                    help="If set, only process chunk_id of the train set. Range: [0, num_chunks).")
    ap.add_argument("--num_chunks", type=int, default=4)
    ap.add_argument("--merge_chunks", action="store_true",
                    help="Merge previously-saved chunk files into the final train_out.")
    # If using the max200f / val_200_frames pre-subsampled annotations, set this so
    # we read frame ids from file_names instead of iterating 1..vid_len.
    ap.add_argument("--use_file_names", action="store_true",
                    help="Read sampled frame indices from `file_names` (for max200f variants).")
    args = ap.parse_args()

    if args.merge_chunks:
        merge_chunks(args.train_out, args.num_chunks)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading CLIP model from {args.model_path} (device={device})")
    model, preprocess = clip.load(args.model_path, device=device)
    for _, param in model.named_parameters():
        param.requires_grad = False

    if not args.skip_train:
        save_split(args.train_ann, args.video_dir, args.train_out, model, preprocess, device,
                   sample_all=not args.use_file_names,
                   chunk_id=args.chunk_id, num_chunks=args.num_chunks)
    if not args.skip_val:
        save_split(args.val_ann, args.video_dir, args.val_out, model, preprocess, device,
                   sample_all=not args.use_file_names)


if __name__ == "__main__":
    main()
