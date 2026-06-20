# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import json 
import argparse
import os
from tqdm import tqdm

import numpy as np
import cv2
import pycocotools.mask as mask_utils



def box_iou(boxes1, boxes2):
  """Compute box IoU. Boxes in format [l, t, w, h].

  Args:
    boxes1: array in shape n x 4
    boxes2: array in shape m x 4
  Returns:
    iou: array in shape n x m
    union: array in shape n x m
  """
#   print("boxes1: ", boxes1)
#   print("boxes2: ", boxes2)
  wh1 = boxes1[:, 2:]
  wh2 = boxes2[:, 2:]
  area1 = wh1[:, 0] * wh1[:, 1]  # [n]
  area2 = wh2[:, 0] * wh2[:, 1]  # [m]
  lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [n, m, 2]
  rb = np.minimum(
      boxes1[:, None, 2:] + boxes1[:, None, :2],
      boxes2[None, :, 2:] + boxes2[None, :, :2])  # [n, m, 2]
  wh = (rb - lt).clip(0.0)  # [n, m, 2]
  intersection = wh[:, :, 0] * wh[:, :, 1]  # [n, m]
  union = area1[:, None] + area2[None, :] - intersection  # [n, m]
  iou = np.where(union > 0, intersection / union, 0)
  return iou


parser = argparse.ArgumentParser()
parser.add_argument('--gt_json', type=str, default='./datasets/VidSTG/annotations/vidstg_max200f_val_coco_format.json')
parser.add_argument('--dt_json', type=str, required=True) # no default
parser.add_argument('-verbose', action='store_true', help='verbose mode')
args = parser.parse_args()

path_gt = args.gt_json
path_pred = args.dt_json

path_out = os.path.join(os.path.dirname(path_pred), 'pred_results/', 'caption_results.json')
# fake_summary = os.path.join(os.path.dirname(path_pred), 'pred_results/', 'summary_results.json')
# fake_relation = os.path.join(os.path.dirname(path_pred), 'pred_results/', 'relation_results.json')

os.makedirs(os.path.dirname(path_out), exist_ok=True)

print("loading gt and pred json files")
with open(path_gt, 'r') as f:
    lvvis_gt = json.load(f)
with open(path_pred, 'r') as f:
    lvvis_pred = json.load(f)

video_ids = list(set([d['video_id'] for d in lvvis_pred]))
# image_ids = [img['id'] for img in coco_gt['images'] if img['video_id'] in video_ids]
print("Saving results for {} videos to SMOT format".format(len(video_ids)))

full_cap_res = {}

img_res = []
total_added_count = 0
for vid in tqdm(video_ids):
    results = [d for d in lvvis_pred if d['video_id'] == vid]
    # print("video_id: {}, results: {}".format(vid, len(results)))
    if len(results) == 0:
        continue
    gts = [d for d in lvvis_gt['annotations'] if d['video_id'] == vid]

    if len(gts) == 0:
        print("No ground truth for video_id: {}".format(vid))
        continue
    
    pred_cap = []
    pred_gts = []
    for r in results:
        # print("Processing result: ", r)
        # loop through gts, compute iou mean over all frames
        bbox= r['bbox']
        # print("length of bbox: ", len(bbox))
        iou_scores = []
        for gt in gts:
            # print("Processing gt: ", gt)
            gt_bbox = gt['bbox']
            vid_iou=[]
            for frame in range(len(gt_bbox)):
                gt_bbox_frame = gt_bbox[frame]
                dt_box_frame = bbox[frame]
                if dt_box_frame is None :
                    continue
                if gt_bbox_frame is None:
                    continue
                iou = box_iou(np.array([dt_box_frame]), np.array([gt_bbox_frame]))[0][0]
                # print("Frame: {}, IoU: {}".format(frame, iou))
                vid_iou.append(iou)
            # print("iou scores :", vid_iou)
            if len(vid_iou) == 0:
                iou_scores.append(0)
            else:
                iou_scores.append(np.mean(vid_iou))
        # find the max iou score
        max_iou = np.max(iou_scores)
        max_iou_idx = np.argmax(iou_scores)
        # print("IoU scores: ", iou_scores)
        # print("Max IoU: {}, Index: {}".format(max_iou, max_iou_idx))
        pred_cap.append(r['caption'][0] if isinstance(r['caption'], list) else r['caption'])
        pred_gts.append(gts[max_iou_idx]['caption'])
        # print("Predicted caption: {}, Ground truth caption: {}".format(r['caption'], gts[max_iou_idx]['caption']))
        # print("\n")
    # save the results
    full_cap_res[str(vid)] = {
        'pred': pred_cap,
        'gt': pred_gts
    }
    # print("full_cap_res[{}]: {}".format(vid, full_cap_res[str(vid)]))
    total_added_count += len(pred_cap)
    # raise NotImplementedError("Saving results to SMOT format is not implemented yet")
with open(path_out, 'w') as f:
    json.dump(full_cap_res, f, indent=4)
print("Saved {} captions to {}".format(total_added_count, path_out))