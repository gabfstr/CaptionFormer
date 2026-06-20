# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""[EXAMPLE] Multi-GPU CLIP image-feature extraction for the VidSTG train set.

This is an *example* — paths are hardcoded and the encode-vs-gather flow is
illustrative rather than production-grade. For most users, the single-GPU
script `save_clip_features.py` (in this same dir) with `--chunk_id` is
sufficient: launch N jobs each with a different chunk id and run
`--merge_chunks` at the end.

Use this file as a starting point if you want to extract on a single node
with N GPUs in parallel. Edit paths + launch with
`python save_clip_features_multigpu_example.py`.
"""
import torch
from clip import clip
from PIL import Image
from tqdm import tqdm
import json
import os
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Function to save clip features
def save_clip_features(rank, world_size, model, preprocess, file_dir, data):
    dic = {}
    # Split the data across the GPUs
    chunk_size = len(data['videos']) // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank != world_size - 1 else len(data['videos'])

    # Process the chunk of videos for this GPU
    for video in tqdm(data['videos'][start_idx:end_idx]):
        video_file_name = video["file_name"].split(".mp4")[0]
        vid_len = video["length"]
        sample_frames = list(range(1, vid_len+1))
        vid_file_names = [os.path.join(file_dir, video_file_name, f"{frame_idx:04d}.jpg") for frame_idx in sample_frames]
        
        for image in vid_file_names:
            file_name = image
            image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(rank)
            feature_clip = model.encode_image(image_clip)
            dic[file_name] = feature_clip.cpu().numpy()  # Move to CPU to avoid GPU memory issues
    return dic

# Main function to set up the parallel process
def main(rank, world_size):
    setup_distributed(rank, world_size)

    model_path = "ViT-B/32"  # auto-downloads to ~/.cache/clip/
    if rank==0:
        print("Loading model from ", model_path)
    
    model, preprocess = clip.load(model_path, device=rank)

    # Use DataParallel to distribute model across multiple GPUs
    model = torch.nn.DataParallel(model)

    for _, param in model.named_parameters():
        param.requires_grad = False

    # VidSTG train
    if rank == 0:
        print("Saving VidSTG train clip features")
    json_path = './datasets/VidSTG/annotations/train_instances_.json'
    file_dir = './datasets/VidSTG/video/'
    save_path = "datasets/metadata/vidstg_train_clip_feature.pkl"
    data = json.load(open(json_path, 'r'))

    data_dic = save_clip_features(rank, world_size, model, preprocess, file_dir, data)
    # Use a simple reduce strategy to gather the results from all GPUs
    dic_all = {}
    torch.distributed.barrier()  # Synchronize all GPUs
    torch.distributed.gather(data_dic, dst=0)  # Gather all results on GPU 0

    if rank == 0:
        for result in data_dic:
            dic_all.update(result)
        torch.save(dic_all, save_path)
        print(f"Features saved to {save_path}")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)