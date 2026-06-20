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
#   from https://github.com/google-research/scenic/blob/main/scenic/projects/densevoc/tools/create_coco_json_from_tfrecord.py

r"""Create coco format json files for evaluation from tfrecords.

This scripts create COCO-format annotation jsons from a TFrecord.
The json is used for mAP and CHOTA evaluation.

Before running this script, please follow the instructions in
`tools/build_vidstg_tfrecord.py` and `tools/build_vln_tfrecord.py` to build
the video TFrecord.

```
mkdir ~/Datasets/VidSTG/annotations

python create_coco_json_from_tfrecord.py -- \
--input_tfrecord ~/Datasets/VidSTG/tfrecords/vidstg.video.max200f.caption.val.tfrecord@32 \
--output_json ~/Datasets/VidSTG/annotations/vidstg_max200f_val_coco_format.json

mkdir ~/Datasets/VLN/annotations

python create_coco_json_from_tfrecord.py -- \
--input_tfrecord ~/Datasets/VLN/tfrecords/vng_uvo_sparse_val.tfrecord@32 \
--output_json  ~/Datasets/VLN/annotations/vng_uvo_sparse_val_coco_format.json
```

"""

import json

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.io import gfile

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tools.data_prep import _input_utils as input_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tfrecord', '', 'path to the tfrecord data.')
flags.DEFINE_string('output_json', '', '')
# flags.DEFINE_string('output_image_path', '', '')
# option to do VLN or VidSTG
flags.DEFINE_bool('vln', False, 'If true, use VLN categories')
flags.DEFINE_bool('smit', False, 'If true, use S-MiT categories')



VIDSTG_CATEGORIES = [
    {'id': 1, 'name': 'adult', 'partition': 1}, {'id': 2, 'name': 'aircraft', 'partition': 1}, 
    {'id': 3, 'name': 'antelope', 'partition': 1}, {'id': 4, 'name': 'baby', 'partition': 1}, 
    {'id': 5, 'name': 'baby_seat', 'partition': 1}, {'id': 6, 'name': 'baby_walker', 'partition': 1}, 
    {'id': 7, 'name': 'backpack', 'partition': 1}, {'id': 8, 'name': 'ball/sports_ball', 'partition': 1}, 
    {'id': 9, 'name': 'bat', 'partition': 1}, {'id': 10, 'name': 'bear', 'partition': 1}, 
    {'id': 11, 'name': 'bench', 'partition': 1}, {'id': 12, 'name': 'bicycle', 'partition': 1}, 
    {'id': 13, 'name': 'bird', 'partition': 1}, {'id': 14, 'name': 'bottle', 'partition': 1}, 
    {'id': 15, 'name': 'bread', 'partition': 1}, {'id': 16, 'name': 'bus/truck', 'partition': 1}, 
    {'id': 17, 'name': 'cake', 'partition': 1}, {'id': 18, 'name': 'camel', 'partition': 1}, 
    {'id': 19, 'name': 'camera', 'partition': 1}, {'id': 20, 'name': 'car', 'partition': 1}, 
    {'id': 21, 'name': 'cat', 'partition': 1}, {'id': 22, 'name': 'cattle/cow', 'partition': 1}, 
    {'id': 23, 'name': 'cellphone', 'partition': 1}, {'id': 24, 'name': 'chair', 'partition': 1}, 
    {'id': 25, 'name': 'chicken', 'partition': 1}, {'id': 26, 'name': 'child', 'partition': 1}, 
    {'id': 27, 'name': 'crab', 'partition': 1}, {'id': 28, 'name': 'crocodile', 'partition': 1}, 
    {'id': 29, 'name': 'cup', 'partition': 1}, {'id': 30, 'name': 'dish', 'partition': 1}, 
    {'id': 31, 'name': 'dog', 'partition': 1}, {'id': 32, 'name': 'duck', 'partition': 1}, 
    {'id': 33, 'name': 'electric_fan', 'partition': 1}, {'id': 34, 'name': 'elephant', 'partition': 1}, 
    {'id': 35, 'name': 'faucet', 'partition': 1}, {'id': 36, 'name': 'fish', 'partition': 1}, 
    {'id': 37, 'name': 'frisbee', 'partition': 1}, {'id': 38, 'name': 'fruits', 'partition': 1}, 
    {'id': 39, 'name': 'guitar', 'partition': 1}, {'id': 40, 'name': 'hamster/rat', 'partition': 1}, 
    {'id': 41, 'name': 'handbag', 'partition': 1}, {'id': 42, 'name': 'horse', 'partition': 1}, 
    {'id': 43, 'name': 'kangaroo', 'partition': 1}, {'id': 44, 'name': 'laptop', 'partition': 1}, 
    {'id': 45, 'name': 'leopard', 'partition': 1}, {'id': 46, 'name': 'lion', 'partition': 1}, 
    {'id': 47, 'name': 'microwave', 'partition': 1}, {'id': 48, 'name': 'motorcycle', 'partition': 1}, 
    {'id': 49, 'name': 'oven', 'partition': 1}, {'id': 50, 'name': 'panda', 'partition': 1}, 
    {'id': 51, 'name': 'penguin', 'partition': 1}, {'id': 52, 'name': 'piano', 'partition': 1}, 
    {'id': 53, 'name': 'pig', 'partition': 1}, {'id': 54, 'name': 'rabbit', 'partition': 1}, 
    {'id': 55, 'name': 'racket', 'partition': 1}, {'id': 56, 'name': 'refrigerator', 'partition': 1}, 
    {'id': 57, 'name': 'scooter', 'partition': 1}, {'id': 58, 'name': 'screen/monitor', 'partition': 1}, 
    {'id': 59, 'name': 'sheep/goat', 'partition': 1}, {'id': 60, 'name': 'sink', 'partition': 1}, 
    {'id': 61, 'name': 'skateboard', 'partition': 1}, {'id': 62, 'name': 'ski', 'partition': 1}, 
    {'id': 63, 'name': 'snake', 'partition': 1}, {'id': 64, 'name': 'snowboard', 'partition': 1}, 
    {'id': 65, 'name': 'sofa', 'partition': 1}, {'id': 66, 'name': 'squirrel', 'partition': 1}, 
    {'id': 67, 'name': 'stingray', 'partition': 1}, {'id': 68, 'name': 'stool', 'partition': 1}, 
    {'id': 69, 'name': 'stop_sign', 'partition': 1}, {'id': 70, 'name': 'suitcase', 'partition': 1}, 
    {'id': 71, 'name': 'surfboard', 'partition': 1}, {'id': 72, 'name': 'table', 'partition': 1}, 
    {'id': 73, 'name': 'tiger', 'partition': 1}, {'id': 74, 'name': 'toilet', 'partition': 1}, 
    {'id': 75, 'name': 'toy', 'partition': 1}, {'id': 76, 'name': 'traffic_light', 'partition': 1}, 
    {'id': 77, 'name': 'train', 'partition': 1}, {'id': 78, 'name': 'turtle', 'partition': 1}, 
    {'id': 79, 'name': 'vegetables', 'partition': 1}, {'id': 80, 'name': 'watercraft', 'partition': 1}
]

VLN_CATEGORIES = [
    {'id': 1, 'name': 'object', 'partition': 1},
]


def decode_sharded_names(path):
  """Convert sharded file names into a list."""
  ret = []
  path = path.split(',')
  for name in path:
    if '@' in name:
      num_shards = int(name.split('@')[1].split('.')[0])
      suffix = name.split(f'@{num_shards}')[-1]
      prefix = name.split('@')[0]
      names = [
          f'{prefix}-{i:05d}-of-{num_shards:05d}{suffix}'
          for i in range(num_shards)
      ]
      ret.extend(names)
    else:
      ret.append(name)
  return ret


def main(unused_argv):
  ds = tf.data.TFRecordDataset(decode_sharded_names(FLAGS.input_tfrecord))
  use_vln = FLAGS.vln
  use_smit = FLAGS.smit
  if use_vln:
    seq_feature_descr = input_utils.densecap_sequence_feature_description_w_msk
  else:
    seq_feature_descr = input_utils.densecap_sequence_feature_description
  if use_smit:
    context_feature_description = input_utils.densecap_context_feature_description_smit
  else:
    context_feature_description = input_utils.densecap_context_feature_description
  ds = ds.map(
      lambda x: tf.io.parse_sequence_example(  # pylint: disable=g-long-lambda
          x,
          sequence_features=seq_feature_descr,
          context_features=context_feature_description))
  ds = ds.map(
      lambda x, y, _:  # pylint: disable=g-long-lambda
      input_utils.decode_and_sample_video_example(
          x, y, _, num_frames=-1, temporal_stride=1, use_category=not(use_vln or use_smit), use_masks=use_vln))
  # print("ds:", ds)
  data_iter = iter(ds)

  video_infos = []
  if use_vln or use_smit:
    cat_info = VLN_CATEGORIES
    category_map = {a['name']: a['id'] for a in VLN_CATEGORIES}
  else:
    cat_info = VIDSTG_CATEGORIES
    category_map = {a['name']: a['id'] for a in VIDSTG_CATEGORIES}
  annotations = []
  ann_count = 0
  num_videos = 0
  no_cat_count = 0
  while True:
    try:
      num_videos += 1
      if num_videos % 100 == 0:
        print(num_videos)
      data = next(data_iter)
    #   print("data keys:", data.keys())
    #   print("video captions:", data['video_captions'])
    #   print("data_path:", data['data_path'])
    # #   print("data images :", data['images'])
    #   print("len(data['images']) :", len(data['images']))
    #   raise ValueError("Debugging")
    except StopIteration:
      break
    if len(video_infos) % 1000 == 0:
      print(f'processed {len(video_infos)} images.')
    images = data['images'].numpy()
    image_ids = data['image_ids'].numpy()
    num_frames, height, width = images.shape[:3]
    video_boxes = data['boxes'].numpy()
    video_track_ids = data['track_ids'].numpy()
    video_captions = data['captions'].numpy()
    video_categories = data['categories'].numpy()
    if use_vln:
      fps = -1  # VLN does not have fps
      video_segmentations = data['segmentations']  # tf.RaggedTensor of shape [T, N]
      decoded_masks = []
      for frame_masks in video_segmentations:  # frame_masks: tf.Tensor of shape [N]
          masks = []
          for rle_bytes in frame_masks.numpy():  # each rle_bytes is a tf.string
              rle_str = rle_bytes.decode('utf-8')
              if rle_str == '':
                  masks.append(None)
              else:
                  rle = json.loads(rle_str)  # Now rle is a dict: {'size': [...], 'counts': '...'}
                  masks.append(rle)
          decoded_masks.append(masks)
    else:
      fps = float(data['fps'])
    if not use_vln:
      og_frame_count = int(data['frame_count']) if 'frame_count' in data else num_frames
    else:
      og_frame_count = 3
    frame_count = len(image_ids)
    # print("og_frame_count:", og_frame_count)
    # print("frame_count:", frame_count)
    # print("len video_track_ids:", len(video_track_ids))
    data_path = data['data_path'].numpy()[0].decode('utf-8')
    frame_ids = data['frame_ids'].numpy()
    try:
      video_id = int(data['video_id'])
    except ValueError:
      video_id = int(num_videos)

    file_name = data_path.replace('.json', '.mp4')

    data_folder = data_path.split('.json')[0]
    # file_names = [f'{data_folder}/{(fid+1):04d}.jpg' for fid in frame_ids]
    if use_vln :
      file_names = [os.path.join(data_folder, str(fid)+'.png') for fid in frame_ids]
    else :
      file_names = [os.path.join(data_folder, f'{(fid+1):04d}.jpg') for fid in frame_ids]
    video_infos.append({
        'file_name': file_name,
        'height': height,
        'width': width,
        'length': frame_count,
        'fps': fps,
        'file_names': file_names,
        'id': int(video_id),
        'og_length': og_frame_count,
    })
    vid_annos = {}
    #Scan video track ids
    for i in range(num_frames):
      track_ids = video_track_ids[i]
      for track_id in track_ids:
        tid = int(track_id)
        if tid!=0 and tid not in vid_annos:
            vid_annos[tid] = {
                'id': tid,
                'video_id': int(video_id),
                'iscrowd': 0,
                'height': height,
                'width': width,
                'length': frame_count,
                'bbox': [None]*frame_count,
                'areas': [0]*frame_count,
                'category_id': -1,
                'caption': ['']*frame_count,
            }
            if use_vln:
              vid_annos[tid]['segmentations'] = [None] * frame_count
    # Add boxes and captions
    for i in range(num_frames):
      boxes = video_boxes[i]
      if use_vln:
        masks = decoded_masks[i]
      else:
        masks = [None] * len(boxes)  # No masks in VidSTG
      phrases = video_captions[i]
      cat_names = video_categories[i]      
      track_ids = video_track_ids[i]
    #   print("categories:", cat_names)
    #   print("boxes:", boxes)
    #   print("phrases:", phrases)
    #   print("track_ids:", track_ids)
    #   from time import sleep
    #   sleep(1)
    #   raise ValueError("Debugging")
      for box, mask, phrase, cat, track_id in zip(boxes, masks, phrases, cat_names, track_ids):
        if box.max() == 0:
          break
        bbox = [
            int(box[0]), int(box[1]),
            int(box[2] - box[0]), int(box[3] - box[1])]
        tid = int(track_id)
        vid_annos[tid]['bbox'][i] = bbox
        vid_annos[tid]['areas'][i] = int(bbox[2] * bbox[3])
        if use_vln:
          vid_annos[tid]['segmentations'][i] = mask if mask is not None else None
        phrase_str = phrase.decode('utf-8')
        if phrase_str != '':
            vid_annos[tid]['caption'][i] = phrase_str
        # phrase_str = phrase.decode('utf-8')
        # if phrase_str != '':
        #     if vid_annos[tid]['caption'] != '':
        #         assert vid_annos[tid]['caption'] == phrase_str, print("vid_annos[tid]['caption']:", vid_annos[tid]['caption'], "phrase_str:", phrase_str)
        #     else:
        #         vid_annos[tid]['caption'] = phrase_str
        cat_str = cat.decode('utf-8')
        # print("category:", cat_str)
        if cat_str != '':
            cat_id = category_map[cat_str]
            # print("cat_id:", cat_id)
            # print("not empty cat_id")
            if vid_annos[tid]['category_id'] != -1:    
                assert vid_annos[tid]['category_id'] == cat_id
            else :
                vid_annos[tid]['category_id'] = cat_id
        
        # ann_count += 1
        # vid_annos[tid] = {
        #     'id': tid,
        #     'video_id': int(video_id),
        #     'iscrowd': 0,
        #     'height': height,
        #     'width': width,
        #     'length': frame_count,
        #     'bbox': [bbox],
        #     'areas': [bbox[2] * bbox[3]],
        #     'category_id': cat_id,
        #     'caption': phrase.decode('utf-8'),
        # }
    # print("\n\n")
    # print("final annos for this video :", vid_annos)
    # print("\n\n")
    # sleep(1)
    # print("vid_annos:", vid_annos)
    # raise ValueError("Debugging")
    for _, vid_anno in vid_annos.items():
    #   if vid_annos[tid]['id']!=37503:
    #   assert vid_anno['category_id'] != -1, print("vid anno :", vid_anno)
      if vid_anno['category_id'] == -1:
        no_cat_count += 1
        # print("no category id for video:", vid_anno)
      assert len(vid_anno['bbox']) == vid_anno['length']
      annotations.append(vid_anno)
    # print("videos:", video_infos)
    # print("categories:", cat_info)
    # print("annotations:", annotations)
    # raise ValueError("Debugging")

  ret = {
      'videos': video_infos,
      'categories': cat_info,
      'annotations': annotations}
  for k, v in ret.items():
    print(k, len(v))
  print('no category count:', no_cat_count)
  json.dump(ret, gfile.GFile(FLAGS.output_json, 'w'))

if __name__ == '__main__':
  app.run(main)