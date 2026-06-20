# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/google-research/scenic/blob/main/scenic/projects/densevoc/chota.py

r"""CHOTA evaluation metric.

CHOTA adds a caption accuracy (CapA) term to the HOTA metric.

This evaluation script assumes the annotations and the predictions are saved
in json in COCO detection-format (https://cocodataset.org/#format-data),
with two additonal keys for both ground turth and predictions:
  'track_id': int; the track identity.
  'caption': string; the object caption.

The code is based on the HOTA metric implemented in
https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/\
metrics/hota.py

Usage:

import json
from chota import CHOTA

gt_data = json.load(open('path/to/coco/format/gt.json', 'r'))
pred_data = json.load(open('path/to/coco/format/pred.json', 'r'))
chota_evaluator = CHOTA()
results = chota_evaluator.compute_metrics(gt_data, pred_data)
print(results)

"""

import re
import cv2
from absl import logging
import numpy as np
from pycocoevalcap.cider import cider
from pycocoevalcap.spice import spice
from pycocoevalcap.meteor import meteor
from pycocotools import mask as mask_utils  # Import the RLE handling utility from pycocotools
import scipy
from tqdm import tqdm


def box_iou(boxes1, boxes2):
  """Compute box IoU. Boxes in format [l, t, w, h].

  Args:
    boxes1: array in shape n x 4
    boxes2: array in shape m x 4
  Returns:
    iou: array in shape n x m
    union: array in shape n x m
  """
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

def mask_iou(masks1, masks2):
    """Compute mask IoU. Masks in format [h, w].

    Args:
      masks1: array in shape n x h x w
      masks2: array in shape m x h x w
    Returns:
      iou: array in shape n x m
      union: array in shape n x m
    """
    area1 = np.sum(masks1, axis=(1, 2))  # [n]
    area2 = np.sum(masks2, axis=(1, 2))  # [m]
    intersection = np.sum(masks1[:, None, :, :] * masks2[None, :, :, :], axis=(2, 3))  # [n, m]
    union = area1[:, None] + area2[None, :] - intersection  # [n, m]
    iou = np.where(union > 0, intersection / union, 0)
    return iou

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
    if area == 0.0 :
        return None
    return box


class CHOTA(object):
  """Class which implements the CHOTA metrics.

  Attribute:
    iou_thresh: list of floats; iou threshold to decide the true-positive or
      false-positive. The overall results will be the average of all thresholds.
    caption_metric: list of strings; caption similarity evaluation metrics. The
      overall CapA will be the average of all metrics.
  """

  def __init__(
      self,
      iou_thresh=(0.5,),  # np.arange(0.05, 0.99, 0.05)
      caption_metric=('cider', 'meteor', 'spice'),
      mask2box=False):
    super().__init__()
    self.array_labels = np.asarray(iou_thresh).reshape(-1)
    self.integer_array_fields = [
        'HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'TP_with_caps']
    self.float_array_fields = [
        'HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA',
        'OWTA', 'CapA']
    self.float_fields = []
    self.fields = self.float_array_fields + self.integer_array_fields + (
        self.float_fields)
    self.summary_fields = self.float_array_fields + self.float_fields
    self.caption_metric = (caption_metric,) if isinstance(
        caption_metric, str) else caption_metric
    print('self.caption_metric', self.caption_metric)
    self.caption_scorer = []
    for caption_metric in self.caption_metric:
      if caption_metric == 'meteor':
        self.caption_scorer.append(meteor.Meteor())
      elif caption_metric == 'cider':
        self.caption_scorer.append(cider.Cider())
      elif caption_metric == 'spice':
        self.caption_scorer.append(spice.Spice())
      else:
        assert caption_metric == 'none', self.caption_metric
    self.mask2box = mask2box

  def compute_caption_similarity(self, gt_captions, pred_captions, i=0):
    """Compute caption metrics."""
    if len(gt_captions) == 0:  # pylint: disable=g-explicit-length-test
      return 0.0, np.ones((0,), dtype=np.float32)
    references = {
        i: [x] for i, x in enumerate(gt_captions)}
    candidate = {
        i: [x] for i, x in enumerate(pred_captions)}
    # print("references", references)
    # print("candidate", candidate)
    try :
      _, ret = self.caption_scorer[i].compute_score(references, candidate)
    except Exception as e:
      print("Error in computing caption similarity for gt_captions and pred_captions :")
      print("gt_captions :", gt_captions)
      print("pred_captions :", pred_captions)
      return 0.0, np.ones((0,), dtype=np.float32)
    # If SPICE upstream returns list[dict] (per-image category breakdowns),
    # reduce to flat float list — different SPICE versions return different
    # shapes and downstream np.sum(ret) needs a flat numeric sequence.
    if len(ret) > 0 and isinstance(ret[0], dict):
      ret = [d['All']['f'] for d in ret]
    # Return sum, will normalize by number of positive instances later.
    return np.sum(ret), ret

  def sanitize_cap(self, gt_captions, dt_captions, max_len=200):
    """Sanitize captions to max len of 200."""
    # print("sanitizing captions")
    # print("captions :", captions)
    sanitized_gts = []
    sanitized_dts = []
    for gt_cap, dt_cap in zip(gt_captions, dt_captions):
      if isinstance(gt_cap, list) and isinstance(dt_cap, list):
        # c_list = [c[:200] for c in cap]
        sanitized_gts.append([self.clean_caption(gt) for gt, dt in zip(gt_cap, dt_cap) if self.is_well_formed(gt) and self.is_well_formed(dt)])
        sanitized_dts.append([self.clean_caption(dt) for gt, dt in zip(gt_cap, dt_cap) if self.is_well_formed(gt) and self.is_well_formed(dt)])
      else:
        raise ValueError(f"Unsupported caption format fot gt and dt captions : {type(gt_cap)}, {type(dt_cap)}")
    return sanitized_gts, sanitized_dts
  
  def is_well_formed(self, sentence, max_len=200):
    rm_captions = [
       "A third man wearing a black suit, is standing in the middle of the first and second man while holding a microphone in his left hand and speaking on the mic while pointing his right hand to the first man",
       "Another girl, whose upper half-body is visible wearing a black t-shirt is sitting on the left side, holding the carrot and specs in her right hand, and then wearing the specs with her hand over her own eyes while speaking",
       "A person whose hand is only visible, tossing meat loaf's inside a black pan",
       "A man whose half body is visible wearing a blue t-shirt is putting pizza into the oven",
       "Another woman whose upper half-body is visible wearing a blue top is standing behind the podium on the stage and is speaking on the mic"
    ]
    for bad_cap in rm_captions:
      if bad_cap in sentence:
        print("sentence :", sentence)
        return False
    return bool(sentence) and sentence.strip()[-1] in '.?!' and len(sentence.split()) > 3

  def clean_caption(self, caption):
    caption = caption.strip()
    # End caption at first '.'
    caption = caption.split('.')[0] + '.'
    caption = re.sub(r'[^\x00-\x7F]+', '', caption)  # remove non-ascii
    caption = re.sub(r'\s+', ' ', caption)
    return caption

  def eval_sequence(self, data):
    """Calculates the HOTA metrics for one sequence.

    Args:
      data: dict with keys:
        'num_tracker_dets': int
        'num_gt_dets': int
        'num_tracker_ids': int
        'num_gt_ids': int
        'gt_ids': list of arrays with length T, each in G_t
        'tracker_ids': list of arrays with length T, each in P_t
        'similarity_scores': list of arrays with length T, in (G_t, P_t)
    Returns:
      res: dict of array
    """
    # print("data :", data)
    # Initialise results
    res = {}
    for field in self.float_array_fields + self.integer_array_fields:
      res[field] = np.zeros((len(self.array_labels)), dtype=np.float32)

    # Return result quickly if tracker or gt sequence is empty
    if data['num_tracker_dets'] == 0:
      res['HOTA_FN'] = data['num_gt_dets'] * np.ones(
          (len(self.array_labels)), dtype=np.float32)
      res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float32)
      for cap_metric in self.caption_metric:
        res[f'CapA-{cap_metric}'] = np.ones(
            (len(self.array_labels)), dtype=np.float32)
      return res
    if data['num_gt_dets'] == 0:
      res['HOTA_FP'] = data['num_tracker_dets'] * np.ones(
          (len(self.array_labels)), dtype=np.float32)
      res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float32)
      for cap_metric in self.caption_metric:
        res[f'CapA-{cap_metric}'] = np.ones(
            (len(self.array_labels)), dtype=np.float32)
      return res

    # Variables counting global association
    potential_matches_count = np.zeros(
        (data['num_gt_ids'], data['num_tracker_ids']))
    gt_id_count = np.zeros((data['num_gt_ids'], 1))
    tracker_id_count = np.zeros((1, data['num_tracker_ids']))

    # First loop through each timestep and accumulate global track information.
    for t, (gt_ids_t, tracker_ids_t) in enumerate(
        zip(data['gt_ids'], data['tracker_ids'])):
      # Count the potential matches between ids in each timestep
      # These are normalised, weighted by the match similarity.
      similarity = data['similarity_scores'][t]
      sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[
          :, np.newaxis] - similarity
      sim_iou = np.zeros_like(similarity)
      sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
      sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[
          sim_iou_mask]
      potential_matches_count[
          gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

      # Calculate the total number of dets for each gt_id and tracker_id.
      gt_id_count[gt_ids_t] += 1
      tracker_id_count[0, tracker_ids_t] += 1

    # Calculate overall jaccard alignment score (before matching) between IDs
    global_alignment_score = potential_matches_count / (
        gt_id_count + tracker_id_count - potential_matches_count)
    matches_counts = [
        np.zeros_like(potential_matches_count) for _ in self.array_labels]

    matched_gt_captions = [[] for _ in self.array_labels]
    matched_pred_captions = [[] for _ in self.array_labels]
    # Calculate scores for each timestep
    for t, (gt_ids_t, tracker_ids_t) in enumerate(
        zip(data['gt_ids'], data['tracker_ids'])):
      # Deal with the case that there are no gt_det/tracker_det in a timestep.
      if len(gt_ids_t) == 0:  # pylint: disable=g-explicit-length-test
        for a, unused_alpha in enumerate(self.array_labels):
          res['HOTA_FP'][a] += len(tracker_ids_t)
        continue
      if len(tracker_ids_t) == 0:  # pylint: disable=g-explicit-length-test
        for a, unused_alpha in enumerate(self.array_labels):
          res['HOTA_FN'][a] += len(gt_ids_t)
        continue

      # Get matching scores between pairs of dets for optimizing HOTA
      similarity = data['similarity_scores'][t]
      score_mat = global_alignment_score[
          gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

      # Hungarian algorithm to find best matches
      match_rows, match_cols = scipy.optimize.linear_sum_assignment(-score_mat)
      gt_captions = data['gt_captions'][t]
      pred_captions = data['pred_captions'][t]
      # Calculate and accumulate basic statistics
      for a, alpha in enumerate(self.array_labels):
        actually_matched_mask = similarity[
            match_rows, match_cols] >= alpha - np.finfo('float').eps
        alpha_match_rows = match_rows[actually_matched_mask]
        alpha_match_cols = match_cols[actually_matched_mask]
        num_matches = len(alpha_match_rows)
        res['HOTA_TP'][a] += num_matches
        res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
        res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
        if num_matches > 0:
          res['LocA'][a] += sum(
              similarity[alpha_match_rows, alpha_match_cols])
          matches_counts[a][
              gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        if self.caption_metric != ('none',) and num_matches > 0:
          matched_gt_caption = [
              gt_captions[x] for x in alpha_match_rows if gt_captions[x]]
          matched_gt_captions[a].extend(matched_gt_caption)
          matched_pred_caption = [
              pred_captions[x] for x, y in zip(
                  alpha_match_cols, alpha_match_rows) if gt_captions[y]]
          matched_pred_captions[a].extend(matched_pred_caption)
          res['TP_with_caps'][a] += len(matched_gt_caption)
    # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
    # First calculate scores per gt_id/tracker_id combo and then average over
    # the number of detections.
    for a, unused_alpha in enumerate(self.array_labels):
      matches_count = matches_counts[a]
      ass_a = matches_count / np.maximum(
          1, gt_id_count + tracker_id_count - matches_count)
      res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(
          1, res['HOTA_TP'][a])
      ass_re = matches_count / np.maximum(1, gt_id_count)
      res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(
          1, res['HOTA_TP'][a])
      ass_pr = matches_count / np.maximum(1, tracker_id_count)
      res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(
          1, res['HOTA_TP'][a])

    # Calculate final scores
    res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(
        1e-10, res['HOTA_TP'])
    for cap_metric_i, cap_metric in enumerate(self.caption_metric):
      if cap_metric=='none':
        continue
      # if cap_metric=='spice':
      #    matched_gt_captions, matched_pred_captions = self.sanitize_cap(matched_gt_captions, matched_pred_captions)
      # print("Computing {} metric".format(str(cap_metric)))
      res[f'CapA-{cap_metric}'] = np.asarray([
          self.compute_caption_similarity(
              matched_gt_captions_a, matched_pred_captions_a, cap_metric_i)[0]
          for (matched_gt_captions_a, matched_pred_captions_a) in zip(
              matched_gt_captions, matched_pred_captions)]) / np.maximum(
                  1e-10, res['TP_with_caps'])
    res = self._compute_final_fields(res)
    return res

  def combine_sequences(self, all_res):
    """Combines metrics across all sequences.

    Args:
      all_res: dict of dict; video_id to res from eval_sequence.
    Returns:
      res: same format as eval_sequence.
    """
    res = {}
    for field in self.integer_array_fields:
      res[field] = self._combine_sum(all_res, field)
    for field in ['AssRe', 'AssPr', 'AssA']:
      res[field] = self._combine_weighted_av(
          all_res, field, res, weight_field='HOTA_TP')
    loca_weighted_sum = sum(
        [all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
    res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(
        1e-10, res['HOTA_TP'])
    for cap_metric in self.caption_metric:
      if cap_metric == 'none':
        continue
      capa_weighted_sum = sum(
          [all_res[k][f'CapA-{cap_metric}'] * all_res[k][
              'TP_with_caps'] for k in all_res.keys()])
      res[f'CapA-{cap_metric}'] = np.maximum(
          1e-10, capa_weighted_sum) / np.maximum(1e-10, res['TP_with_caps'])
    res = self._compute_final_fields(res, caption_metrics=self.caption_metric)
    return res

  @staticmethod
  def _compute_final_fields(res, caption_metrics=()):
    """Calculate sub-metric values which only depend on other values."""
    res['DetRe'] = res['HOTA_TP'] / np.maximum(
        1, res['HOTA_TP'] + res['HOTA_FN'])
    res['DetPr'] = res['HOTA_TP'] / np.maximum(
        1, res['HOTA_TP'] + res['HOTA_FP'])
    res['DetA'] = res['HOTA_TP'] / np.maximum(
        1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
    res['OWTA'] = np.sqrt(res['DetRe'] * res['AssA'])
    res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])

    if len(caption_metrics) == 3:
      res['CapA'] = 0
      for cap_metric in caption_metrics:
        res['CapA'] += res[f'CapA-{cap_metric}'] / len(caption_metrics)
      res['CHOTA'] = (res['DetA'] * res['AssA'] * res['CapA']) ** (1./ 3.)
    return res

  @staticmethod
  def _combine_sum(all_res, field):
    """Combine sequence results via sum."""
    return sum([all_res[k][field] for k in all_res.keys()])

  @staticmethod
  def _combine_weighted_av(all_res, field, comb_res, weight_field):
    """Combine sequence results via weighted average."""
    return sum(
        [all_res[k][field] * all_res[k][weight_field] for k in all_res.keys()]
    ) / np.maximum(1.0, comb_res[weight_field])

  @staticmethod
  def convert_coco_to_hota_format(
      gt_data, pred_data, score_thresh=-1., caption_metric='none'):
    """Convert coco format to HOTA required format.

    Args:
      gt_data: coco json format with key "annotations" and "images".
      pred_data: coco prediction format, list of dict in "annotation" format.
      score_thresh: float; convert score-based detection to hard detection.
      caption_metric: str; 'none', 'cider', 'meteor'.
    Returns:
      Dict of dict; video_id to data that will be used in eval_sequence.
    """

    dt_key = 'bbox' if 'bbox' in pred_data[0] else 'segmentations'

    all_ret = {}
    imageid2videoid = {x['id']: x['video_id'] for x in gt_data['images']}
    video_ids = set(imageid2videoid.values())
    videoid2size = { x['video_id']: (x['height'], x['width']) for x in gt_data['images']}
    # print("len of pred_data", len(pred_data))
    # print("threshold", score_thresh)
    pred_data = [x for x in pred_data if x['score'] > score_thresh]
    # print("after filtering pred_data", len(pred_data))
    pred_data_video = {video_id: [] for video_id in video_ids}
    # print("pred_data_video", pred_data_video)
    # raise NotImplementedError("pred_data_video is not implemented yet")
    gt_data_video = {video_id: [] for video_id in video_ids}
    for x in pred_data:
      pred_data_video[imageid2videoid[x['image_id']]].append(x)
    for x in gt_data['annotations']:
      gt_data_video[imageid2videoid[x['image_id']]].append(x)
    from tqdm import tqdm
    for video_id in tqdm(video_ids):
      ret = {}
      image_ids = sorted(set(x['id'] for x in gt_data['images']
                             if imageid2videoid[x['id']] == video_id))
      # print("image_ids", image_ids)
      vid_h, vid_w = videoid2size[video_id]
      ret['num_tracker_dets'] = len(pred_data_video[video_id])
      ret['num_gt_dets'] = len(gt_data_video[video_id])
      tracker_ids = set(x['track_id'] for x in pred_data_video[video_id])
      gt_ids = set(x['track_id'] for x in gt_data_video[video_id])
      ret['num_tracker_ids'] = len(tracker_ids)
      ret['num_gt_ids'] = len(gt_ids)
      pred_id_map = {v: k for k, v in enumerate(sorted(tracker_ids))}
      gt_id_map = {v: k for k, v in enumerate(sorted(gt_ids))}
      id2gts = {x: [] for x in image_ids}
      id2preds = {x: [] for x in image_ids}
      for x in gt_data_video[video_id]:
        id2gts[x['image_id']].append(x)
      for x in pred_data_video[video_id]:
        id2preds[x['image_id']].append(x)
      gt_ids, tracker_ids, similarity_scores = [], [], []
      gt_captions, pred_captions = [], []
      for test_i, image_id in enumerate(image_ids):
        gt_id = np.asarray(
            [gt_id_map[x['track_id']] for x in id2gts[image_id]],
            dtype=np.int32).reshape(-1)
        tracker_id = np.asarray(
            [pred_id_map[x['track_id']] for x in id2preds[image_id]],
            dtype=np.int32).reshape(-1)
        if dt_key == 'segmentations':
            try :
              gt_masks = np.asarray(
                decode_rle([x['segmentation'] for x in id2gts[image_id]]),
                dtype=np.uint8).transpose(2, 0, 1).reshape(-1, vid_h, vid_w)
            except Exception as e:
              print("error in decoding rle")
              print("video_id", video_id)
              print("image_id", image_id)
              print("id2gts[image_id]", id2gts[image_id])
              print("id2preds[image_id]", id2preds[image_id])
              print(('id2gts[image_id] :', id2gts[image_id]))
              print([x['segmentation'] for x in id2gts[image_id]])
              raise e
            # print("gt_masks shape", gt_masks.shape)
            # print("number of non 0 pixels", np.sum(gt_masks))
            tracker_masks = np.asarray(
              decode_rle([x['segmentation'] for x in id2preds[image_id]]),
              dtype=np.uint8).transpose(2, 0, 1).reshape(-1, vid_h, vid_w)
            
            similarity_scores.append(mask_iou(gt_masks, tracker_masks))
            
        else :
          gt_boxes = np.asarray(
              [x['bbox'] for x in id2gts[image_id]],
              dtype=np.float32).reshape(-1, 4)
          # print("[x['bbox'] for x in id2preds[image_id]] :", [x['bbox'] for x in id2preds[image_id]])
          tracker_boxes = np.asarray(
              [x['bbox'] for x in id2preds[image_id]],
              dtype=np.float32).reshape(-1, 4)
          # print("similarity_scores", box_iou(gt_boxes, tracker_boxes))
          similarity_scores.append(box_iou(gt_boxes, tracker_boxes))
        gt_ids.append(gt_id)
        tracker_ids.append(tracker_id)

        # if caption_metric != 'none' and caption_metric != ('none',):
        gt_captions.append([x['caption'] if x.get('caption') else '' for x in id2gts[image_id]])
        pred_captions.append([x['caption'].split('\n')[0] if x.get('caption') else '' for x in id2preds[image_id]])
          # print("pred_captions", pred_captions)
          # raise NotImplementedError("caption metric is not implemented yet")
      ret['gt_ids'] = gt_ids
      ret['tracker_ids'] = tracker_ids
      ret['similarity_scores'] = similarity_scores
      ret['gt_captions'] = gt_captions
      ret['pred_captions'] = pred_captions
      all_ret[video_id] = ret
    return all_ret
  
  @staticmethod
  def convert_lvvis_to_hota_format(gt_data, pred_data, score_thresh=-1., caption_metric='none', mask2box=False):
    """Convert LVVIS video format to HOTA required format.

    Args:
      gt_data: Dict with "videos" and "annotations" (LVVIS format).
      pred_data: List of predictions per video (LVVIS adapted format).
      score_thresh: float; filter out low-confidence predictions.
      caption_metric: str; 'none', 'cider', 'meteor'.

    Returns:
      Dict of dict; video_id to data that will be used in eval_sequence.
    """
    all_ret = {}
    
    # Get video lengths
    video_lengths = {v['id']: v['length'] for v in gt_data['videos']}

    # Filter out low-score predictions
    pred_data = [x for x in pred_data if x['score'] > score_thresh]

    # Organize GT and predictions per video
    pred_data_video = {v_id: [] for v_id in video_lengths}
    gt_data_video = {v_id: [] for v_id in video_lengths}

    pred_key = 'bbox' if 'bbox' in pred_data[0] else 'segmentations'
    for x in pred_data:
        pred_data_video[x['video_id']].append(x)
    gt_key = 'bbox' if 'bbox' in gt_data['annotations'][0] else 'segmentations'
    for x in gt_data['annotations']:
        gt_data_video[x['video_id']].append(x)


    # first_vid_id= gt_data['videos'][0]['id']
    # gt_key = 'bbox' if 'bbox' in gt_data_video[first_vid_id][0] else 'segmentations'
    # pred_key = 'bbox' if 'bbox' in pred_data_video[first_vid_id][0] else 'segmentations'
    print("Converting gt data from key {} and pred data from key {}: ".format(gt_key, pred_key))

    # Process each video
    for video_id, num_frames in tqdm(video_lengths.items()):
        ret = {}

        # Number of detections
        ret['num_tracker_dets'] = sum(len([b for b in x[pred_key] if b]) for x in pred_data_video[video_id])
        ret['num_gt_dets'] = sum(len([b for b in x[gt_key] if b]) for x in gt_data_video[video_id])

        ret['num_tracker_ids'] = len(pred_data_video[video_id])
        ret['num_gt_ids'] = len(gt_data_video[video_id])

        # Sort gt objects by id
        gt_data_video[video_id] = sorted(gt_data_video[video_id], key=lambda x: x['id'])

        gt_ids_list, tracker_ids_list, similarity_scores_list = [], [], []
        gt_captions_list, pred_captions_list = [], []


        # Iterate over frames
        for frame_idx in range(num_frames):
            gt_objects = [x for x in gt_data_video[video_id] if frame_idx < len(x[gt_key])]
            pred_objects = [x for x in pred_data_video[video_id] if frame_idx < len(x[pred_key])]

            gt_boxes_or_masks = []
            gt_id = []
            if len(gt_objects)>0 :
              h,w = gt_objects[0]["height"], gt_objects[0]["width"]
            else :
              h,w = 224,224
            for track_id, x in enumerate(gt_objects):
                if frame_idx < len(x[gt_key]) and x[gt_key][frame_idx] is not None:
                    if gt_key == 'segmentations':
                        mask = mask_utils.decode(rle)
                        mask = (mask * 255).astype(np.uint8)
                        if mask2box:
                          bbox = _get_bbox_from_mask(mask)
                          gt_boxes_or_masks.append(bbox)
                        else :
                          gt_boxes_or_masks.append(mask)
                    else:
                        # assert mask2box == True
                        gt_boxes_or_masks.append(x['bbox'][frame_idx])
                    gt_id.append(track_id)
            # print("gt_boxes", gt_boxes)
            # print("len gt_boxes", len(gt_boxes))
            if mask2box or gt_key == 'bbox':
              gt_boxes_or_masks = np.array(gt_boxes_or_masks, dtype=np.float32).reshape(-1, 4)
            else :
              # h,w = gt_objects[0][gt_key][frame_idx]['size'] if len(gt_objects)>0 else (0,0)
              # print("h,w", h,w)
              gt_boxes_or_masks = np.array(gt_boxes_or_masks, dtype=np.uint8).reshape(-1, h, w)
            gt_id = np.array(gt_id, dtype=np.int32).reshape(-1)
            
            pred_boxes_or_masks = []
            tracker_id = []
            for track_id, x in enumerate(pred_objects):
                if frame_idx < len(x[pred_key]) and x[pred_key][frame_idx] is not None:
                    if pred_key == 'segmentations':
                        rle = {
                            'counts': x[pred_key][frame_idx]['counts'],
                            'size': x[pred_key][frame_idx]['size']
                        }
                        mask = mask_utils.decode(rle)
                        mask = (mask * 255).astype(np.uint8)
                        if mask2box:
                          bbox = _get_bbox_from_mask(mask)
                          if bbox is None:
                            continue
                          pred_boxes_or_masks.append(bbox)
                        else :
                          pred_boxes_or_masks.append(mask)
                          h, w = mask.shape[-2], mask.shape[-1]
                    else:
                        # assert mask2box == True
                        pred_boxes_or_masks.append(x['bbox'][frame_idx])
                    tracker_id.append(track_id)
            # print("pred_boxes", pred_boxes)
            # print("len pred_boxes", len(pred_boxes))
            if mask2box or pred_key == 'bbox':
              pred_boxes_or_masks = np.array(pred_boxes_or_masks, dtype=np.float32).reshape(-1, 4) 
            else :
              pred_boxes_or_masks = np.array(pred_boxes_or_masks, dtype=np.uint8).reshape(-1, h, w)
            tracker_id = np.array(tracker_id, dtype=np.int32).reshape(-1)

            
            # print("gt_boxes", gt_boxes.shape)
            # print("pred_boxes", pred_boxes.shape)
            if mask2box or pred_key == 'bbox':
              similarity = box_iou(gt_boxes_or_masks, pred_boxes_or_masks) if len(gt_boxes_or_masks) > 0 and len(pred_boxes_or_masks) > 0 else np.zeros((len(gt_boxes_or_masks), len(pred_boxes_or_masks)))
            else:
              similarity = mask_iou(gt_boxes_or_masks, pred_boxes_or_masks) if len(gt_boxes_or_masks) > 0 and len(pred_boxes_or_masks) > 0 else np.zeros((len(gt_boxes_or_masks), len(pred_boxes_or_masks)))
            # print("similarity", similarity)
            gt_ids_list.append(gt_id)
            tracker_ids_list.append(tracker_id)
            similarity_scores_list.append(similarity)

            if caption_metric != 'none':
                gt_captions_list.append([x.get('caption', '')[frame_idx] if isinstance(x.get('caption', ''), list) else x.get('caption', '') for x in gt_objects])
                pred_captions_list.append([x.get('caption', '')[0] if isinstance(x.get('caption', ''), list) else x.get('caption', '') for x in pred_objects])
            
            # Temp 
            for i in range(len(pred_captions_list)):
                cap = pred_captions_list[i]
                if cap is not None :
                    pred_captions_list[i] = [x.split("\n")[0] for x in cap if x is not None]
            
        ret['gt_ids'] = gt_ids_list
        ret['tracker_ids'] = tracker_ids_list
        ret['similarity_scores'] = similarity_scores_list
        ret['gt_captions'] = gt_captions_list
        ret['pred_captions'] = pred_captions_list
        all_ret[video_id] = ret

    return all_ret

  @staticmethod
  def convert_old_lvvis_to_hota_format(gt_data, pred_data, score_thresh=-1., caption_metric='none', mask2box=False):
    """Convert LVVIS video format to HOTA required format.

    Args:
      gt_data: Dict with "videos" and "annotations" (LVVIS format).
      pred_data: List of predictions per video (LVVIS adapted format).
      score_thresh: float; filter out low-confidence predictions.
      caption_metric: str; 'none', 'cider', 'meteor'.

    Returns:
      Dict of dict; video_id to data that will be used in eval_sequence.
    """
    all_ret = {}
    
    # Get video lengths
    video_lengths = {v['id']: v['length'] for v in gt_data['videos']}

    # Filter out low-score predictions
    pred_data = [x for x in pred_data if x['score'] > score_thresh]

    # Organize GT and predictions per video
    pred_data_video = {v_id: [] for v_id in video_lengths}
    gt_data_video = {v_id: [] for v_id in video_lengths}

    pred_key = 'bbox' if 'bbox' in pred_data[0] else 'segmentations'
    for x in pred_data:
        pred_data_video[x['video_id']].append(x)
    gt_key = 'bbox' if 'bbox' in gt_data['annotations'][0] else 'segmentations'
    for x in gt_data['annotations']:
        gt_data_video[x['video_id']].append(x)


    # first_vid_id= gt_data['videos'][0]['id']
    # gt_key = 'bbox' if 'bbox' in gt_data_video[first_vid_id][0] else 'segmentations'
    # pred_key = 'bbox' if 'bbox' in pred_data_video[first_vid_id][0] else 'segmentations'
    print("Converting gt data from key {} and pred data from key {}: ".format(gt_key, pred_key))

    # Process each video
    for video_id, num_frames in tqdm(video_lengths.items()):
        ret = {}

        # Number of detections
        ret['num_tracker_dets'] = sum(len([b for b in x[pred_key] if b]) for x in pred_data_video[video_id])
        ret['num_gt_dets'] = sum(len([b for b in x[gt_key] if b]) for x in gt_data_video[video_id])

        ret['num_tracker_ids'] = len(pred_data_video[video_id])
        ret['num_gt_ids'] = len(gt_data_video[video_id])

        # Sort gt objects by id
        gt_data_video[video_id] = sorted(gt_data_video[video_id], key=lambda x: x['id'])

        gt_ids_list, tracker_ids_list, similarity_scores_list = [], [], []
        gt_captions_list, pred_captions_list = [], []


        # Iterate over frames
        for frame_idx in range(num_frames):
            gt_objects = [x for x in gt_data_video[video_id] if frame_idx < len(x[gt_key])]
            pred_objects = [x for x in pred_data_video[video_id] if frame_idx < len(x[pred_key])]

            gt_boxes_or_masks = []
            gt_id = []
            if len(gt_objects)>0 :
              h,w = gt_objects[0]["height"], gt_objects[0]["width"]
            else :
              h,w = 224,224
            for track_id, x in enumerate(gt_objects):
                if frame_idx < len(x[gt_key]) and x[gt_key][frame_idx] is not None:
                    if gt_key == 'segmentations':
                        rle = {
                            'counts': x[gt_key][frame_idx]['counts'],
                            'size': x[gt_key][frame_idx]['size']
                        }
                        mask = mask_utils.decode(rle)
                        mask = (mask * 255).astype(np.uint8)
                        if mask2box:
                          bbox = _get_bbox_from_mask(mask)
                          gt_boxes_or_masks.append(bbox)
                        else :
                          gt_boxes_or_masks.append(mask)
                    else:
                        # assert mask2box == True
                        gt_boxes_or_masks.append(x['bbox'][frame_idx])
                    gt_id.append(track_id)
            # print("gt_boxes", gt_boxes)
            # print("len gt_boxes", len(gt_boxes))
            if mask2box or gt_key == 'bbox':
              gt_boxes_or_masks = np.array(gt_boxes_or_masks, dtype=np.float32).reshape(-1, 4)
            else :
              # h,w = gt_objects[0][gt_key][frame_idx]['size'] if len(gt_objects)>0 else (0,0)
              # print("h,w", h,w)
              gt_boxes_or_masks = np.array(gt_boxes_or_masks, dtype=np.uint8).reshape(-1, h, w)
            gt_id = np.array(gt_id, dtype=np.int32).reshape(-1)
            
            pred_boxes_or_masks = []
            tracker_id = []
            for track_id, x in enumerate(pred_objects):
                if frame_idx < len(x[pred_key]) and x[pred_key][frame_idx] is not None:
                    if pred_key == 'segmentations':
                        rle = {
                            'counts': x[pred_key][frame_idx]['counts'],
                            'size': x[pred_key][frame_idx]['size']
                        }
                        mask = mask_utils.decode(rle)
                        mask = (mask * 255).astype(np.uint8)
                        if mask2box:
                          bbox = _get_bbox_from_mask(mask)
                          if bbox is None:
                            continue
                          pred_boxes_or_masks.append(bbox)
                        else :
                          pred_boxes_or_masks.append(mask)
                          h, w = mask.shape[-2], mask.shape[-1]
                    else:
                        # assert mask2box == True
                        pred_boxes_or_masks.append(x['bbox'][frame_idx])
                    tracker_id.append(track_id)
            # print("pred_boxes", pred_boxes)
            # print("len pred_boxes", len(pred_boxes))
            if mask2box or pred_key == 'bbox':
              pred_boxes_or_masks = np.array(pred_boxes_or_masks, dtype=np.float32).reshape(-1, 4) 
            else :
              pred_boxes_or_masks = np.array(pred_boxes_or_masks, dtype=np.uint8).reshape(-1, h, w)
            tracker_id = np.array(tracker_id, dtype=np.int32).reshape(-1)

            
            # print("gt_boxes", gt_boxes.shape)
            # print("pred_boxes", pred_boxes.shape)
            if mask2box or pred_key == 'bbox':
              similarity = box_iou(gt_boxes_or_masks, pred_boxes_or_masks) if len(gt_boxes_or_masks) > 0 and len(pred_boxes_or_masks) > 0 else np.zeros((len(gt_boxes_or_masks), len(pred_boxes_or_masks)))
            else:
              similarity = mask_iou(gt_boxes_or_masks, pred_boxes_or_masks) if len(gt_boxes_or_masks) > 0 and len(pred_boxes_or_masks) > 0 else np.zeros((len(gt_boxes_or_masks), len(pred_boxes_or_masks)))
            # print("similarity", similarity)
            gt_ids_list.append(gt_id)
            tracker_ids_list.append(tracker_id)
            similarity_scores_list.append(similarity)

            if caption_metric != 'none':
                gt_captions_list.append([x.get('caption', '')[0] if isinstance(x.get('caption', ''), list) else x.get('caption', '') for x in gt_objects])
                pred_captions_list.append([x.get('caption', '')[0] if isinstance(x.get('caption', ''), list) else x.get('caption', '') for x in pred_objects])
            
            # Temp 
            for i in range(len(pred_captions_list)):
                cap = pred_captions_list[i]
                if cap is not None :
                    pred_captions_list[i] = [x.split("\n")[0] for x in cap if x is not None]
            
        ret['gt_ids'] = gt_ids_list
        ret['tracker_ids'] = tracker_ids_list
        ret['similarity_scores'] = similarity_scores_list
        ret['gt_captions'] = gt_captions_list
        ret['pred_captions'] = pred_captions_list
        all_ret[video_id] = ret

    return all_ret
  
  @staticmethod
  def convert_img_lvis_to_hota_format(gt_data, pred_data, score_thresh=-1., caption_metric='none'):
    """ For testing purposes, convert image predictions to HOTA format to compute framewise DetA and LocA.

    Args : 
    gt_data : Dict with "images" and "annotations" (Lvis format).
    pred_data : List of predictions per image (lvis results format).
    score_thresh : float; filter out low-confidence predictions.
    caption_metric : str; 'none', 'cider', 'meteor'.
    """
      
    all_ret = {}
  
    # Get video lengths
    video_lengths = {v['id']: 1 for v in gt_data['images']}

    # Filter out low-score predictions
    pred_data = [x for x in pred_data if x['score'] > score_thresh]

    # Organize GT and predictions per video
    pred_data_video = {v_id: [] for v_id in video_lengths}
    gt_data_video = {v_id: [] for v_id in video_lengths}

    pred_key = 'bbox' if 'bbox' in pred_data[0] else 'segmentations'
    for x in pred_data:
        pred_data_video[x['image_id']].append(x)
    gt_key = 'bbox' if 'bbox' in gt_data['annotations'][0] else 'segmentations'
    for x in gt_data['annotations']:
        gt_data_video[x['image_id']].append(x)


    # first_vid_id= gt_data['videos'][0]['id']
    # gt_key = 'bbox' if 'bbox' in gt_data_video[first_vid_id][0] else 'segmentations'
    # pred_key = 'bbox' if 'bbox' in pred_data_video[first_vid_id][0] else 'segmentations'
    print("Converting gt data from key {} and pred data from key {}: ".format(gt_key, pred_key))

    # Process each video
    for video_id, num_frames in tqdm(video_lengths.items()):
        ret = {}

        # Number of detections
        ret['num_tracker_dets'] = len(pred_data_video[video_id])
        ret['num_gt_dets'] = len(gt_data_video[video_id])

        ret['num_tracker_ids'] = len(pred_data_video[video_id])
        ret['num_gt_ids'] = len(gt_data_video[video_id])

        # Sort gt objects by id
        gt_data_video[video_id] = sorted(gt_data_video[video_id], key=lambda x: x['id'])

        gt_ids_list, tracker_ids_list, similarity_scores_list = [], [], []
        gt_captions_list, pred_captions_list = [], []


        
        gt_objects = [x for x in gt_data_video[video_id]]
        pred_objects = [x for x in pred_data_video[video_id]]

        gt_boxes = []
        gt_id = []
        for track_id, x in enumerate(gt_objects):   
          if gt_key == 'segmentations':
              rle = {
                  'counts': x[gt_key]['counts'],
                  'size': x[gt_key]['size']
              }
              mask = mask_utils.decode(rle)
              mask = (mask * 255).astype(np.uint8)
              bbox = _get_bbox_from_mask(mask)
              gt_boxes.append(bbox)
          else:
              gt_boxes.append(x['bbox'])
          gt_id.append(track_id)
        # print("gt_boxes", gt_boxes)
        # print("len gt_boxes", len(gt_boxes))
        gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 4)
        gt_id = np.array(gt_id, dtype=np.int32).reshape(-1)
        
        pred_boxes = []
        tracker_id = []
        for track_id, x in enumerate(pred_objects):
          if pred_key == 'segmentations':
              rle = {
                  'counts': x[pred_key]['counts'],
                  'size': x[pred_key]['size']
              }
              mask = mask_utils.decode(rle)
              mask = (mask * 255).astype(np.uint8)
              bbox = _get_bbox_from_mask(mask)
              if bbox is None:
                continue
              pred_boxes.append(bbox)
          else:
              pred_boxes.append(x['bbox'])
          tracker_id.append(track_id)
        # print("pred_boxes", pred_boxes)
        # print("len pred_boxes", len(pred_boxes))
        pred_boxes = np.array(pred_boxes, dtype=np.float32).reshape(-1, 4) 
        tracker_id = np.array(tracker_id, dtype=np.int32).reshape(-1)

        
        # print("gt_boxes", gt_boxes.shape)
        # print("pred_boxes", pred_boxes.shape)
        similarity = box_iou(gt_boxes, pred_boxes) if len(gt_boxes) > 0 and len(pred_boxes) > 0 else np.zeros((len(gt_boxes), len(pred_boxes)))
        # print("similarity", similarity)
        # raise ValueError("Stop")
        gt_ids_list.append(gt_id)
        tracker_ids_list.append(tracker_id)
        similarity_scores_list.append(similarity)

        if caption_metric != 'none':
            gt_captions_list.append([x.get('caption', '')[0] if isinstance(x.get('caption', ''), list) else x.get('caption', '') for x in gt_objects])
            pred_captions_list.append([x.get('caption', '')[0] if isinstance(x.get('caption', ''), list) else x.get('caption', '') for x in pred_objects])
        
        # Temp 
        for i in range(len(pred_captions_list)):
            cap = pred_captions_list[i]
            if cap is not None :
                pred_captions_list[i] = [x.split("\n")[0] for x in cap if x is not None]
            
        ret['gt_ids'] = gt_ids_list
        ret['tracker_ids'] = tracker_ids_list
        ret['similarity_scores'] = similarity_scores_list
        ret['gt_captions'] = gt_captions_list
        ret['pred_captions'] = pred_captions_list
        all_ret[video_id] = ret

    return all_ret

  def compute_metrics(
      self, gt_data, pred_data, score_thresh=0.5, ann_format='coco' ,image=False):
    """Compute HOTA on coco format.

    Args:
      gt_data: coco json format with key "annotations" and "images".
      pred_data: coco prediction format, list of dict in "annotation" format.
      score_thresh: float; convert score-based detection to hard detection.
      image: bool; if True, compute framewise DetA, LocA ...
    Returns:
      Dict of floats; evaluation results.
    """
    logging.info('Converting format...')
    assert ann_format in ['coco', 'lvvis'], 'Unsupported annotation format'
    if ann_format == 'coco':
        eval_data = self.convert_coco_to_hota_format(
            gt_data, pred_data, score_thresh=score_thresh,
            caption_metric=self.caption_metric)
    else :
      if image:
        eval_data = self.convert_img_lvis_to_hota_format(
            gt_data, pred_data, score_thresh=score_thresh,
            caption_metric=self.caption_metric)
      else:
        eval_data = self.convert_lvvis_to_hota_format(
            gt_data, pred_data, score_thresh=score_thresh,
            caption_metric=self.caption_metric, mask2box=self.mask2box)
    logging.info('Evaluating sequences...')
    sequence_results = {}
    for i, (k, v) in tqdm(enumerate(eval_data.items()), total=len(eval_data)):
      if i % 100 == 0:
        logging.info('%d of %d', i, len(eval_data))
      sequence_results[k] = self.eval_sequence(v)
    logging.info('Combining results...')
    final_results = self.combine_sequences(sequence_results)
    final_results = {
        k: float(v) if isinstance(v, (np.float32, float)) else float(
            v.sum() / len(v))
        for k, v in final_results.items()}
    return final_results