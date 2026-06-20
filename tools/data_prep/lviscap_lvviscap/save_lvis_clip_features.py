# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""Extract CLIP image features for LVIS (or merged LVIS+COCO) splits.

Iterates over the `images` field of a COCO/LVIS-style annotation JSON,
encodes each image with the CLIP ViT-B/32 visual encoder, saves a single
{image_id: feature_tensor} pickle per split.

Run from the repo root. The resulting `.pkl` files are loaded by the model
via `MODEL.MASK_FORMER.CLIP_IMAGE_PATH` during training/inference.
"""
import argparse
import json
import os

import torch
from clip import clip
from PIL import Image
from tqdm import tqdm


def resolve_image_path(image_rec, file_dir):
    """LVIS/COCO annotations may store either `file_name` or `coco_url` —
    handle both. Strips the old `COCO_train2014_*` prefix if present."""
    if "file_name" in image_rec:
        name = image_rec["file_name"]
        if name.startswith("COCO"):
            name = name[-16:]
        return os.path.join(file_dir, name)
    if "coco_url" in image_rec:
        # e.g., http://images.cocodataset.org/train2017/000000391895.jpg
        return os.path.join(file_dir, image_rec["coco_url"][30:])
    raise KeyError(f"image record has neither file_name nor coco_url: keys={list(image_rec)}")


def save_split(json_path, file_dir, save_path, model, preprocess, device, val_fallback_dir=None):
    print(f"\n=== {os.path.basename(save_path)} ===")
    print(f"  ann:  {json_path}")
    print(f"  imgs: {file_dir}")
    print(f"  out:  {save_path}")
    data = json.load(open(json_path, "r"))
    dic = {}
    for image in tqdm(data["images"]):
        path = resolve_image_path(image, file_dir)
        if val_fallback_dir is not None and not os.path.exists(path):
            # Some LVIS val images live in train2017/ — try a fallback.
            path = os.path.join(val_fallback_dir, f"{image['id']:012d}.jpg")
        image_clip = preprocess(Image.open(path)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[image["id"]] = feature_clip
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dic, save_path)
    print(f"  saved {len(dic)} features → {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", default="./datasets/lvis/lvis_v1_train.json",
                    help="Training annotation JSON. Use ./datasets/lvis/lviscap_v1_train+coco.json "
                         "for the LVIS+COCO merged training set.")
    ap.add_argument("--val_ann", default="./datasets/lvis/lvis_v1_val.json")
    ap.add_argument("--train_img_dir", default="./datasets/coco/train2017/")
    ap.add_argument("--val_img_dir", default="./datasets/coco/val2017/")
    ap.add_argument("--train_out", default="./datasets/metadata/lvis_train_clip_feature.pkl",
                    help="For LVIS+COCO merge, override to lvis+coco_train_clip_feature.pkl.")
    ap.add_argument("--val_out", default="./datasets/metadata/lvis_val_clip_feature.pkl")
    ap.add_argument("--model_path", default="ViT-B/32",
                    help="CLIP model name (auto-downloaded to ~/.cache/clip/) "
                         "or path to a .pt file.")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_val", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading CLIP model from {args.model_path} (device={device})")
    model, preprocess = clip.load(args.model_path, device=device)
    for _, param in model.named_parameters():
        param.requires_grad = False

    if not args.skip_train:
        save_split(args.train_ann, args.train_img_dir, args.train_out, model, preprocess, device)
    if not args.skip_val:
        # LVIS val sometimes references train2017 images — pass a fallback dir for that case.
        save_split(args.val_ann, args.val_img_dir, args.val_out, model, preprocess, device,
                   val_fallback_dir=args.train_img_dir)


if __name__ == "__main__":
    main()
