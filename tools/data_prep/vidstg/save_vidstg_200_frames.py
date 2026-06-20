# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import argparse
import json
import random
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/VidSTG/val_instances_.json', help='Path to the input annotations file.')
    parser.add_argument('--num_frames', type=int, default=200, help='Number of frames to sample.')
    args = parser.parse_args()

    print('Loading', args.ann)
    # Load the data
    with open(args.ann, 'r') as f:
        data = json.load(f)

    for vid in tqdm(data['videos']):
        vid_id = vid['id']
        vid_len = vid['length']

        file_name = vid['file_name']
        # If len < num_frames, just give all of them
        if vid_len <= args.num_frames:
            vid['og_length'] = vid_len
            vid['length'] = vid_len
            vid['file_names'] = [file_name.split('.mp4')[0] + f"/{(i+1):04d}.jpg" for i in range(vid_len)]
            continue
        
        if args.num_frames == 2 :
            #Sample a frame from first third of the video and the other from the last third
            sample_ids = [random.randint(0, vid_len//3), random.randint(2*vid_len//3, vid_len-1)]
        else :
            # Sample args.num_frames consistently across videos
            sample_ids = [int(vid_len * i / args.num_frames) for i in range(args.num_frames)]
        # print("sample_ids : ", sample_ids)
        # print("len sample_ids : ", len(sample_ids))
        # print("vid_len : ", vid_len)
        # print("vid_id : ", vid_id)
        assert len(sample_ids) == args.num_frames
        assert sample_ids[-1] < vid_len
        # id_baba+=1
        # if id_baba==6:
        #     raise ValueError("vid_id : ", vid_id)

        vid['og_length'] = vid_len
        vid['length'] = len(sample_ids)

        # file_names = [file_name.split('.mp4')[0] + f"/{(frame_idx+1):04d}.jpg" for frame_idx in sample_ids]
        file_names = [vid['file_names'][i] for i in sample_ids]
        vid['file_names'] = file_names
        # print("sample ids : ", sample_ids)
        # Update annotations
        for ann in data['annotations']:
            if ann['video_id'] == vid_id:
                # print("keys : ", ann.keys())
                ann["og_length"] = vid_len
                ann["length"] = len(sample_ids)
                ann["bbox"] = [ann["bbox"][fid] for fid in sample_ids]
                ann["areas"] = [ann["areas"][fid] for fid in sample_ids]
    
    # Save the filtered dataset
    out_path = args.ann.split('.json')[0].replace('_200_frames','') + '{}_frames_.json'.format(args.num_frames)
    print('Saving to', out_path)
    with open(out_path, 'w') as f:
        json.dump(data, f)