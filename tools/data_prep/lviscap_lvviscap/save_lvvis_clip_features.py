# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""Extract CLIP image features for LV-VIS / LVVIScap splits.

Iterates over the `videos[].file_names` field of a per-track annotation JSON,
encodes each frame with the CLIP ViT-B/32 visual encoder, saves a single
{frame_relative_path: feature_tensor} pickle per split.

Run from the repo root. The resulting `.pkl` files are loaded by the model
via `MODEL.MASK_FORMER.CLIP_IMAGE_PATH` during training/inference. (Note: VLN
and BenSMOT eval configs also point at `lvvis_val_clip_feature.pkl`.)
"""
import argparse
import json
import os

import torch
from clip import clip
from PIL import Image
from tqdm import tqdm


def save_split(json_path, file_dir, save_path, model, preprocess, device):
    print(f"\n=== {os.path.basename(save_path)} ===")
    print(f"  ann:  {json_path}")
    print(f"  imgs: {file_dir}")
    print(f"  out:  {save_path}")
    data = json.load(open(json_path, "r"))
    dic = {}
    for video in tqdm(data["videos"]):
        for image in video["file_names"]:
            path = os.path.join(file_dir, image)
            image_clip = preprocess(Image.open(path)).unsqueeze(0).to(device)
            feature_clip = model.encode_image(image_clip)
            dic[path] = feature_clip
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dic, save_path)
    print(f"  saved {len(dic)} features → {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", default="./datasets/LVVIS/train/train_instances_.json")
    ap.add_argument("--val_ann",   default="./datasets/LVVIS/val/val_instances_.json")
    ap.add_argument("--test_ann",  default="./datasets/LVVIS/test/test_instances.json")
    ap.add_argument("--train_img_dir", default="./datasets/LVVIS/train/JPEGImages/")
    ap.add_argument("--val_img_dir",   default="./datasets/LVVIS/val/JPEGImages/")
    ap.add_argument("--test_img_dir",  default="./datasets/LVVIS/test/JPEGImages/")
    ap.add_argument("--train_out", default="./datasets/metadata/lvvis_train_clip_feature.pkl")
    ap.add_argument("--val_out",   default="./datasets/metadata/lvvis_val_clip_feature.pkl")
    ap.add_argument("--test_out",  default="./datasets/metadata/lvvis_test_clip_feature.pkl")
    ap.add_argument("--model_path", default="ViT-B/32",
                    help="CLIP model name (auto-downloaded to ~/.cache/clip/) "
                         "or path to a .pt file.")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_val", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading CLIP model from {args.model_path} (device={device})")
    model, preprocess = clip.load(args.model_path, device=device)
    for _, param in model.named_parameters():
        param.requires_grad = False

    if not args.skip_train:
        save_split(args.train_ann, args.train_img_dir, args.train_out, model, preprocess, device)
    if not args.skip_val:
        save_split(args.val_ann, args.val_img_dir, args.val_out, model, preprocess, device)
    if not args.skip_test:
        save_split(args.test_ann, args.test_img_dir, args.test_out, model, preprocess, device)


if __name__ == "__main__":
    main()
