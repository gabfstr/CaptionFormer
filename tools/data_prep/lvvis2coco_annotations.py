# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import json 
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gt_json', type=str, default='./datasets/LVVIS/val_instances_.json')
parser.add_argument('--out_path', type=str, default=None,
                    help='output path; default = <gt_json>_coco_format.json (trailing _ stripped)')
parser.add_argument('-box_only', action='store_true', help='use boxes only')
parser.add_argument('-verbose', action='store_true', help='verbose mode')
args = parser.parse_args()


path_gt = args.gt_json

print("loading gt json file")
with open(path_gt, 'r') as f:
    coco_gt = json.load(f)

category_info = coco_gt['categories']

# video_ids = [x['id'] for x in coco_gt['videos']]
video_info = coco_gt['videos']
img_infos = []
annotations = []
total_img_count = 0
total_ann_count = 0
for vid in tqdm(video_info):
    vid_id = vid['id']
    vid_info = [x for x in coco_gt['videos'] if x['id'] == vid]
    num_frames = vid['length']
    vid_annotations = [x for x in coco_gt['annotations'] if x['video_id'] == vid_id]

    if args.verbose:
        print("processing video {} with {} frames".format(vid, num_frames))
        print("Found {} tracks".format(len(vid_annotations)))

    for i in range(num_frames):
        total_img_count += 1
        rec = {
            'file_name': vid['file_names'][i],
            'id': total_img_count,
            'height': vid['height'],
            'width': vid['width'],
            'video_id': vid_id,
        }
        img_infos.append(rec)

        for track_id, track in enumerate(vid_annotations):
            
            # print("track keys :", track.keys())
            # print("segmentations :", track['segmentations'])
            # print("len :", len(track['segmentations']))
            # print("bbox :", track['bbox'])
            # raise Exception("stop here")
            if 'segmentations' in track:
                assert len(track['segmentations']) == num_frames, f"vid {vid_id} has {len(track['segmentations'])} frames but {num_frames} images"
                # assert len(track['bbox']) == num_frames, f"vid {vid_id} has {len(track['bbox'])} frames but {num_frames} images"
                if track['segmentations'][i] is None:
                    if args.box_only != True:
                        continue
            if 'bbox' in track:
                assert len(track['bbox']) == num_frames, f"vid {vid_id} has {len(track['bbox'])} frames but {num_frames} images"
                # assert len(track['segmentations']) == num_frames, f"vid {vid_id} has {len(track['segmentations'])} frames but {num_frames} images"
                if track['bbox'][i] is None:
                    continue

            # if track['bbox'][i] is None:
            #     continue
            # if track['bbox'][i].max() == 0:
            #     continue

            total_ann_count += 1
            if 'segmentations' in track:
                rec = {
                    'id': total_ann_count,
                    'iscrowd': track['iscrowd'],
                    'image_id': total_img_count,
                    'category_id': track['category_id'],
                    'segmentation': track['segmentations'][i],
                    'bbox': track['bbox'][i],
                    'area': track['areas'][i],
                    'caption': track['caption'][i] if isinstance(track['caption'], list) else track['caption'],
                    'track_id': track_id,
                }
            else :
                rec = {
                    'id': total_ann_count,
                    'iscrowd': track['iscrowd'],
                    'image_id': total_img_count,
                    'category_id': track['category_id'],
                    'bbox': track['bbox'][i],
                    'area': track['areas'][i],
                    'caption': track['caption'][i] if isinstance(track['caption'], list) else track['caption'],
                    'track_id': track_id,
                }

            annotations.append(rec)
        
    
    # if args.verbose:
    #     print("Added {} detections in total for video {}\n".format(, vid))
    # from time import sleep
    # sleep(1)
    # raise Exception("stop here")
# raise Exception("stop here")
# print("Total detections number : ", total_added_count)
print("processed {} videos and {} frames".format(len(video_info), total_img_count))
coco_res = {
    'images': img_infos,
    'annotations': annotations,
    'categories': category_info,
}

out_path = args.out_path or (path_gt[:-len('.json')].rstrip('_') + '_coco_format.json')
# Save new results
with open(out_path, 'w') as f:
    json.dump(coco_res, f)
print("Coco format results saved at ", out_path)