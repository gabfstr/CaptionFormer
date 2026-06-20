# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import json
import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import math
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from captionformer.modeling.captioning_head.eva_vit import create_eva_vit_g
from captionformer.modeling.captioning_head.blip2_opt import LayerNorm
from lvis import LVIS

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def main(rank, world_size):
    setup_distributed(rank, world_size)

    
    annotation_path = './datasets/lvis/vg_box_train.json'
    image_root = './datasets/VisualGenome/'
    error_log_file = "./datasets/VisualGenome/features/feature_extraction_errors.txt"
    index_mapping_file = "./datasets/VisualGenome/features/index_mapping.json"

    lvis_api = LVIS(annotation_path)
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)

    if rank == 0:
        print("Initializing visual encoder...")

    img_size = 364

    visual_encoder = create_eva_vit_g(
        img_size, drop_path_rate=0, use_checkpoint=False, precision="fp32"
    ).to(rank)

    visual_encoder = nn.parallel.DistributedDataParallel(visual_encoder, device_ids=[rank])
    visual_encoder.eval()

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if rank == 0:
        print(f"Loaded {len(imgs)} images from {annotation_path}")

    try:
        with open(error_log_file, "r") as f:
            error_files = set(line.strip() for line in f)
    except FileNotFoundError:
        error_files = set()

    # Load mapping
    try:
        with open(index_mapping_file, "r") as f:
            index_mapping = json.load(f)
    except FileNotFoundError:
        index_mapping = {}

    new_errors = set()

    # Split images across ranks
    imgs = imgs[rank::world_size]

    batch_size = 64
    num_batches = math.ceil(len(imgs) / batch_size)

    with tqdm(total=num_batches, desc=f"Rank {rank} Processing Batches") as pbar:
        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, len(imgs))
            batch_imgs = imgs[start_index:end_index]

            # Check folder exists
            new_file_folder = os.path.join(image_root,"features/")
            os.makedirs(new_file_folder, exist_ok=True)
            
            batch_output_filename = os.path.join(new_file_folder, f"batch_{batch_index + 1}_of_{num_batches}.pth")

            batch_features = []
            for idx, img_dict in enumerate(batch_imgs):
                image_id = img_dict["id"]
                if "coco_url" in img_dict:
                    file_name = img_dict["coco_url"].split("cocodataset.org/")[-1]
                else:
                    file_name = img_dict["file_name"]
                img_path = os.path.join(image_root, file_name)

                # Store mapping
                index_mapping[image_id] = {"file_name": batch_output_filename, "index": idx}

                if os.path.exists(batch_output_filename):
                    if batch_output_filename in error_files:
                        print("Processing images from error file:", batch_output_filename)
                        error_files.remove(batch_output_filename)
                        with open(error_log_file, "w") as f:
                            for line in error_files:
                                f.write(line + "\n")
                    else:
                        continue

                
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(rank)
                with torch.no_grad():
                    features = visual_encoder(image).cpu()
                batch_features.append(features)

            if batch_features:
                try :
                    batch_features = torch.cat(batch_features, dim=0)
                    torch.save(batch_features, batch_output_filename)
                    print(f"Rank {rank}: Saved batch features to {batch_output_filename}")
                except Exception as e:
                    print(f"Error processing {img_path} on rank {rank}: {e}")
                    new_errors.add(batch_output_filename)
                    with open(error_log_file, "a") as error_file:
                        error_file.write(f"{batch_output_filename}\n")
            
            # Save mapping
            with open(index_mapping_file, "w") as f:
                json.dump(index_mapping, f)
            
            pbar.update(1)

    if rank == 0:
        print("Finished processing all images. Errors files stored in", error_log_file)

    cleanup_distributed()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
