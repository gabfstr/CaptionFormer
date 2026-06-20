# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import torch
from clip import clip
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ViT-B/32"  # auto-downloads to ~/.cache/clip/
print("loading CLIP model:", model_path)
model, preprocess = clip.load(model_path, device=device)
for _, param in model.named_parameters():
    param.requires_grad = False


VIDSTG_CLASSES = ('adult', 'aircraft', 'antelope', 'baby', 'baby_seat', 'baby_walker', 'backpack', 'ball/sports_ball',
                    'bat', 'bear', 'bench', 'bicycle', 'bird', 'bottle', 'bread', 'bus/truck', 'cake', 'camel', 'camera',
                    'car', 'cat', 'cattle/cow', 'cellphone', 'chair', 'chicken', 'child', 'crab', 'crocodile', 'cup', 'dish',
                    'dog', 'duck', 'electric_fan', 'elephant', 'faucet', 'fish', 'frisbee', 'fruits', 'guitar', 'hamster/rat',
                    'handbag', 'horse', 'kangaroo', 'laptop', 'leopard', 'lion', 'microwave', 'motorcycle', 'oven', 'panda',
                    'penguin', 'piano', 'pig', 'rabbit', 'racket', 'refrigerator', 'scooter', 'screen/monitor', 'sheep/goat',
                    'sink', 'skateboard', 'ski', 'snake', 'snowboard', 'sofa', 'squirrel', 'stingray', 'stool', 'stop_sign',
                    'suitcase', 'surfboard', 'table', 'tiger', 'toilet', 'toy', 'traffic_light', 'train', 'turtle',
                    'vegetables', 'watercraft')


# tokenizer = clip.get_tokenizer("ViT-B-32")

# Create text prompts
text_prompts = [f"A photo of {classname.replace('_', ' ')}" for classname in VIDSTG_CLASSES]

# Tokenize and encode text
text_tokens = clip.tokenize(text_prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# Normalize the embeddings
text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Optional normalization

# Convert to numpy and save
np.save("./datasets/metadata/vidstg_simple_clip_text.npy", text_features.cpu().numpy())
print("Saved text features to vidstg_simple_clip_text.npy")
