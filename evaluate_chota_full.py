# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""
Unified eval entry point: per-track raw `results.json` to CHOTA score, one command.

Per-frame score thresholding are applied inside the
model at inference time.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from tqdm import tqdm


# VLN extended to sparse frames filter 
def filter_per_track_to_sparse(lvvis_pred, extended_gt, *, masks=False):
    video_ids = list(set([d['video_id'] for d in lvvis_pred]))
    new_res = []
    for vid in tqdm(video_ids, desc="vln extended to sparse"):
        vid_info = [x for x in extended_gt['videos'] if x['id'] == vid]
        if len(vid_info) == 0:
            print("video {} not found in extended gt".format(vid))
            continue
        assert len(vid_info) == 1, f"Video {vid} has multiple entries in extended gt"
        og_idx = vid_info[0]["sparse_frame_indices"]
        vid_res = [res for res in lvvis_pred if res['video_id'] == vid]
        for track_res in vid_res:
            rec = {
                "video_id": track_res["video_id"],
                "score": track_res["score"],
                "category_id": track_res["category_id"],
                "caption": track_res["caption"],
            }
            if masks:
                rec['segmentations'] = [a for i, a in enumerate(track_res["segmentations"]) if i in og_idx]
            else:
                rec['bbox'] = [a for i, a in enumerate(track_res["bbox"]) if i in og_idx]
                rec["score_per_frame"] = [a for i, a in enumerate(track_res["score_per_frame"]) if i in og_idx]
            new_res.append(rec)
    return new_res



# lvvis2coco format helpers
def decode_rle(rle):
    if isinstance(rle, dict) or isinstance(rle, list):
        if isinstance(rle, list) and len(rle) == 0:
            return np.zeros((0, 0, 0), dtype=np.uint8)
        mask = mask_utils.decode(rle)
        mask = mask.astype(np.uint8)
        return mask
    raise ValueError("RLE format is not supported")


def _get_bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
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
    if area == 0.0:
        return None
    return box


def convert_per_track_to_coco(lvvis_pred, coco_gt, *, masks=False, mask2box=False, verbose=False, threshold=0.0):
    """Mirrors tools/data_prep/lvvis2coco_results.py. `threshold` defaults to 0
    (no-op — release path applies thresholds at inference); pass >0 to sweep at eval time.
    """
    pred_thresh = threshold
    video_ids = list(set([d['video_id'] for d in lvvis_pred]))

    img_res = []
    total_added_count = 0
    for vid in tqdm(video_ids):
        image_ids = [img['id'] for img in coco_gt['images'] if img['video_id'] == vid]
        if len(image_ids) == 0:
            print("video {} not found in gt".format(vid))
            continue
        vid_res = [res for res in lvvis_pred if res['video_id'] == vid]
        num_frames = len(image_ids)
        if verbose:
            print("processing video {} with {} frames".format(vid, num_frames))
            print("Found {} tracks".format(len(vid_res)))
        vid_added_count = 0
        for track_id, track_res in enumerate(vid_res):
            if masks or mask2box:
                assert len(track_res['segmentations']) == num_frames, \
                    f"vid {vid} has {len(track_res['segmentations'])} frames but {num_frames} images"
            else:
                assert len(track_res['bbox']) == num_frames, \
                    f"vid {vid} has {len(track_res['bbox'])} frames but {num_frames} images"
            for frame_idx in range(num_frames):
                if masks or mask2box:
                    curr_score = track_res['score']
                else:
                    curr_score = track_res['score_per_frame'][frame_idx]
                if curr_score <= pred_thresh:
                    continue
                if masks or mask2box:
                    if track_res['segmentations'][frame_idx] is None:
                        continue
                else:
                    if track_res['bbox'][frame_idx] is None:
                        continue

                vid_added_count += 1
                total_added_count += 1
                cap = track_res['caption']
                if isinstance(cap, list):
                    cap = cap[0] if len(cap) == 1 else cap[frame_idx]
                if masks or mask2box:
                    if mask2box:
                        seg = track_res['segmentations'][frame_idx]
                        mask = decode_rle(seg)
                        box = _get_bbox_from_mask(mask)
                        if box is None:
                            continue
                        img_res.append({
                            'image_id': image_ids[frame_idx],
                            'video_id': vid,
                            'category_id': track_res['category_id'],
                            'score': curr_score,
                            'track_id': track_id,
                            'bbox': box,
                            'caption': cap,
                        })
                    else:
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
                    img_res.append({
                        'image_id': image_ids[frame_idx],
                        'video_id': vid,
                        'bbox': track_res['bbox'][frame_idx],
                        'category_id': track_res['category_id'],
                        'score': curr_score,
                        'track_id': track_id,
                        'caption': cap,
                    })
        if verbose:
            print("Added {} detections in total for video {}".format(vid_added_count, vid))
    print("Total detections number : ", total_added_count)
    return img_res


# CHOTA scoring function
def main():
    parser = argparse.ArgumentParser(description="CHOTA eval (unified: convert + score)")
    parser.add_argument('--gt_json', required=True, help='Ground truth in COCO format (per-image annotations)')
    parser.add_argument('--dt_json', required=True, help='Raw per-track predictions (results.json from inference)')
    parser.add_argument('-no_caption', action='store_true', help='No captioning evaluation')
    parser.add_argument('-no_spice', action='store_true', help='No SPICE evaluation (requires Java)')
    parser.add_argument('-cider_only', action='store_true', help='Evaluate only CIDEr (skip SPICE/METEOR)')
    parser.add_argument('-masks', action='store_true', help='Evaluate using masks instead of bounding boxes for iou matching')
    parser.add_argument('-mask2box', action='store_true', help='Use masks to extract boxes (then eval as boxes)')
    parser.add_argument('-verbose', action='store_true', help='Verbose conversion logs')
    parser.add_argument('--save-intermediate', action='store_true',
                        help='Also write the intermediate _coco_format.json next to dt_json')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Default 0 (thresholds at inference).')
    parser.add_argument('--vln_extended_gt', default=None,
                        help='VLN only: path to the extended GT (with sparse_frame_indices per '
                             'video). When provided, predictions are first filtered to sparse '
                             'frames before COCO conversion + scoring.')
    args = parser.parse_args()

    print("loading gt and pred json files")
    with open(args.gt_json, 'r') as f:
        gt_data = json.load(f)
    with open(args.dt_json, 'r') as f:
        lvvis_pred = json.load(f)

    if args.vln_extended_gt is not None:
        with open(args.vln_extended_gt, 'r') as f:
            extended_gt = json.load(f)
        lvvis_pred = filter_per_track_to_sparse(lvvis_pred, extended_gt, masks=args.masks)

    print("converting per-track predictions to COCO per-image format")
    pred_data = convert_per_track_to_coco(
        lvvis_pred, gt_data,
        masks=args.masks, mask2box=args.mask2box, verbose=args.verbose,
        threshold=args.threshold,
    )
    if args.save_intermediate:
        out_path = args.dt_json.replace('.json', '_coco_format.json')
        with open(out_path, 'w') as f:
            json.dump(pred_data, f)
        print("Coco format intermediate saved at", out_path)

    # CHOTA scoring 
    from evaluate import CHOTA  # noqa: E402
    caption_metrics = ('cider', 'meteor', 'spice')
    if args.no_spice:
        caption_metrics = 'none' if args.no_caption else ('cider', 'meteor')
    if args.cider_only:
        caption_metrics = 'none' if args.no_caption else ('cider',)
    if args.no_caption:
        caption_metrics = 'none'
    chota_evaluator = CHOTA(caption_metric=caption_metrics, mask2box=args.masks)

    results = chota_evaluator.compute_metrics(
        gt_data, pred_data,
        score_thresh=0.0,        # thresholding done at inference
        ann_format='coco',
        image=False,
    )

    print(results)
    # Save next to dt_json
    base = os.path.splitext(args.dt_json)[0]
    suffix = '' if args.threshold == 0.0 else f'_t{int(round(args.threshold * 10)):02d}'
    out_txt = f'{base}_chota_full{suffix}.txt'
    with open(out_txt, 'w') as f:
        f.write(str(results))
    print("Results saved to", out_txt)


if __name__ == "__main__":
    main()
