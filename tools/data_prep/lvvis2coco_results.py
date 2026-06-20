# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import json 
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import pycocotools.mask as mask_utils

parser = argparse.ArgumentParser()
parser.add_argument('--gt_json', type=str, default='./datasets/VidSTG/annotations/vidstg_max200f_val_coco_format.json')
parser.add_argument('--dt_json', type=str, required=True) # no default
parser.add_argument('--threshold', type=float, default=0.4)
parser.add_argument('-masks', action='store_true', help='use masks')
parser.add_argument('-mask2box', action='store_true', help='use masks to get boxes')
parser.add_argument('-verbose', action='store_true', help='verbose mode')
args = parser.parse_args()


def decode_rle(rle):
    """Decode RLE to binary mask.

    Args:
      rle: dict with keys 'counts' and 'size'.
    Returns:
      mask: binary mask in shape (h, w).
    """
    if isinstance(rle, dict) or isinstance(rle, list):
        if isinstance(rle, list) and len(rle) == 0:
            return np.zeros((0, 0, 0), dtype=np.uint8)
        mask = mask_utils.decode(rle)
        mask = mask.astype(np.uint8)
        return mask
    else:
        raise ValueError("RLE format is not supported")
    
def _get_bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) == 0:
    #     print("mask : ", mask)
    #     print("ask type : ", type(mask))
    #     print("sum of pixels : ", np.sum(mask))
    #     return None
    
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    # x_min, y_min = 0.0, 0.0
    # x_max, y_max = 0.0, 0.0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    if x_min == float('inf') or y_min == float('inf') or x_max == float('-inf') or y_max == float('-inf'):
        return None
    
    box = [x_min, y_min, x_max - x_min, y_max - y_min]
    area = (x_max - x_min) * (y_max - y_min)
    if area == 0.0 :
        return None
    return box


# path1 = './datasets/VidSTG/annotations/val_instances_200_frames_.json'
# path2 = './datasets/VidSTG/annotations/vidstg_max200f_val_instances_.json'

path_gt = args.gt_json
path_pred = args.dt_json
pred_thresh = args.threshold

print("loading gt and pred json files")
with open(path_gt, 'r') as f:
    coco_gt = json.load(f)
with open(path_pred, 'r') as f:
    lvvis_pred = json.load(f)

video_ids = list(set([d['video_id'] for d in lvvis_pred]))
# image_ids = [img['id'] for img in coco_gt['images'] if img['video_id'] in video_ids]

img_res = []
total_added_count = 0
for vid in tqdm(video_ids):
    image_ids = [img['id'] for img in coco_gt['images'] if img['video_id'] == vid]
    if len(image_ids) == 0:
        print("video {} not found in gt".format(vid))
        continue
    vid_res = [res for res in lvvis_pred if res['video_id'] == vid]
    num_frames = len(image_ids)
    if args.verbose:
        print("processing video {} with {} frames".format(vid, num_frames))
        print("Found {} tracks".format(len(vid_res)))
    vid_added_count = 0
    for track_id, track_res in enumerate(vid_res):
        # print("track {} with {} frames".format(track_id, len(track_res['bbox'])))
        if args.masks or args.mask2box:
            assert len(track_res['segmentations'])==num_frames, f"vid {vid} has {len(track_res['segmentations'])} frames but {num_frames} images"
        else :
            assert len(track_res['bbox'])==num_frames, f"vid {vid} has {len(vid_res['bbox'])} frames but {num_frames} images"
        count_added = 0
        for frame_idx in range(num_frames):
            if args.masks or args.mask2box:
                curr_score = track_res['score']
            else :
                curr_score = track_res['score_per_frame'][frame_idx]
            if curr_score < pred_thresh:
                continue
            
            # continue if bbox is None
            if args.masks or args.mask2box:
                if track_res['segmentations'][frame_idx] is None:
                    continue
            else :
                if track_res['bbox'][frame_idx] is None:
                    continue
            

            count_added += 1
            vid_added_count += 1
            total_added_count += 1
            if args.masks or args.mask2box:
                if args.mask2box:
                    seg = track_res['segmentations'][frame_idx]
                    mask = decode_rle(seg)
                    box = _get_bbox_from_mask(mask)
                    if box is None:
                        # print("mask shape : ", mask.shape)
                        # print("box is None")
                        continue
                    # print("adding bbox : ", box)
                    cap = track_res['caption']
                    if isinstance(cap, list):
                        cap = cap[0] if len(cap)==1 else cap[frame_idx]
                    img_res.append({
                        'image_id': image_ids[frame_idx],
                        'video_id': vid,
                        'category_id': track_res['category_id'],
                        'score': curr_score,
                        'track_id': track_id,
                        'bbox': box,
                        'caption': cap,
                    })
                else :
                    cap = track_res['caption']
                    if isinstance(cap, list):
                        cap = cap[0] if len(cap)==1 else cap[frame_idx]
                    img_res.append({
                        'image_id': image_ids[frame_idx],
                        'video_id': vid,
                        'category_id': track_res['category_id'],
                        'score': curr_score,
                        'track_id': track_id,
                        'segmentation': track_res['segmentations'][frame_idx],
                        'caption': cap,
                    })
            else:
                cap = track_res['caption']
                if isinstance(cap, list):
                    cap = cap[0] if len(cap)==1 else cap[frame_idx]
                img_res.append({
                    'image_id': image_ids[frame_idx],
                    'video_id': vid,
                    'bbox': track_res['bbox'][frame_idx],
                    'category_id': track_res['category_id'],
                    'score': curr_score,
                    'track_id': track_id,
                    'caption': cap,
                })
        # print("Added {} detections for track {}".format(count_added, track_id))
    if args.verbose:
        print("Added {} detections in total for video {}\n".format(vid_added_count, vid))
    # from time import sleep
    # sleep(1)
    # raise Exception("stop here")

print("Total detections number : ", total_added_count)

out_path = path_pred.replace('.json', '_coco_format.json')
# Save new results
with open(out_path, 'w') as f:
    json.dump(img_res, f)
print("Coco format results saved at ", out_path)