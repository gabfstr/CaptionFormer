# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_json', type=str, default='./datasets/VLN/annotations/vng_uvo_sparse_val_instances_.json')
parser.add_argument('-box_only', action='store_true', help='use masks to get boxes')
args = parser.parse_args()
# ann_path = './datasets/VLN/annotations/vng_uvo_sparse_val_instances_.json'
# ann_path = './datasets/VLN/annotations/vng_uvo_sparse_val_instances_.json'
ann_path = args.gt_json
box_only = args.box_only

out_path   = ann_path.replace('val', 'val_extended')

OFFSETS    = (-25, -20, -15, -10, -5, 0)

with open(ann_path, 'r') as f:
    data = json.load(f)

video_data = data['videos']

vid_id_2_frame_id = {}
vid_id_2_new_length = {}
for video in video_data:
    vid_indices = video['file_names']
    # print("vid indices: ", vid_indices)
    idx = [int(os.path.basename(vid).split('.')[0]) for vid in vid_indices]
    vid_name = vid_indices[0].split('/')[0] 
    # Add previous frames
    extended_idx = set()
    og_frame_idx = []
    pos_counter=0
    for i in idx:
        for off in OFFSETS:
            new_i = i + off
            if new_i >= 0:
                extended_idx.add(new_i)
                pos_counter += 1
        og_frame_idx.append(pos_counter-1)
    vid_id_2_frame_id[video['id']] = og_frame_idx
    extended_idx = sorted(extended_idx)
    # print("Original indices: ", idx)
    # print("Extended indices: ", extended_idx)

    extended_vid_indices = [os.path.join(vid_name, f"{i}.png") for i in extended_idx]
    # print("Extended video indices: ", extended_vid_indices)
    # print("vid_id_2_frame_id: ", vid_id_2_frame_id)
    # raise ValueError("Check extended indices")
    video["file_names"] = extended_vid_indices
    video["length"] = len(extended_vid_indices)
    video["sparse_frame_indices"] = og_frame_idx
    vid_id_2_new_length[video['id']] = len(extended_vid_indices)


ann_data = data['annotations']
# Iterate through the annotations and extend them
for ann in ann_data:
    vid_id = ann['video_id']
    # print("keys in ann: ", ann.keys())
    og_idx = vid_id_2_frame_id[vid_id]
    # print("Original indices in annotation: ", og_idx)
    new_len = vid_id_2_new_length[vid_id]
    # print("New length for video {}: {}".format(vid_id, new_len))
    new_bbox = [None] * new_len
    new_segm = [None] * new_len
    new_area = [None] * new_len
    new_caption = [''] * new_len

    area_list = ann['areas']
    box_list = ann['bbox']
    segm_list = ann['segmentations'] if 'segmentations' in ann else [None] * new_len
    cap_list = ann['caption'] if 'caption' in ann else [''] * len(og_idx)
    # print("cap_list: ", cap_list)
    if isinstance(cap_list, str):
        cap_list = [cap_list] * len(og_idx)
        raise ValueError("cap_list is a string, expected a list of captions")
    frame_added = 0
    for segm, box, area, caption, frame_id in zip(segm_list, box_list, area_list, cap_list, og_idx):
        if segm is not None:
            # print("Segmentation: ", segm, " Frame ID: ", frame_id)
            new_segm[frame_id] = segm
        if box is not None:
            # print("Box: ", box, " Frame ID: ", frame_id)
            new_bbox[frame_id] = box
        if area is not None:
            # print("Area: ", area, " Frame ID: ", frame_id)
            new_area[frame_id] = area
        if caption is not None:
            new_caption[frame_id] = caption
        frame_added += 1
    
    ann['segmentations'] = new_segm
    ann['bbox'] = new_bbox
    ann['areas'] = new_area
    ann['length'] = new_len
    ann["caption"] = new_caption
    # print("new ann: ", ann)
    # raise ValueError("Check new annotations")
    
print("data keys: ", data.keys())
new_data = {
    'categories': data['categories'],
    'videos': video_data,
    'annotations': ann_data,
}
print("length of new annotations: ", len(new_data['annotations']))

with open(out_path, 'w') as f:
    json.dump(new_data, f, indent=4)
print(f"Extended annotation file saved to {out_path}")