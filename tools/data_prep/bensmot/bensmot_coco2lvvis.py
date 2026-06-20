# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import os
import json
import tqdm
import numpy as np
import PIL.Image as Image


DATA_PATH = './datasets/bensmot'
OUT_PATH = './datasets/bensmot/annotations/'
SPLITS = ['train', 'test']

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    with open('./datasets/bensmot/video_summary.json', 'r') as f:
        summary_dict = json.load(f)
    with open('./datasets/bensmot/instance_caption.json', 'r') as f:
        caption_dict = json.load(f)
    with open('./datasets/bensmot/relation.json', 'r') as f:
        relation_dict = json.load(f)
    seqmap = open('./datasets/bensmot/seqmaps/BenSMOT-val.txt', 'w')
    seqmap.write('name\n')
    for split in SPLITS:
        out = {'videos': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        cate_dirs = os.listdir(os.path.join(DATA_PATH, split))
        for cate_dir in tqdm.tqdm(cate_dirs, desc=split + ' set'):
            cate_dir_path = os.path.join(DATA_PATH, split, cate_dir)
            seq_dirs = os.listdir(cate_dir_path)
            for idx, seq_dir in enumerate(seq_dirs):
                seq_dir_path = os.path.join(cate_dir_path, seq_dir)
                seq_name = cate_dir + '/' + seq_dir
                if seq_name not in summary_dict or seq_name not in caption_dict or seq_name not in relation_dict:
                    continue
                # video_summary = summary_dict[seq_name]
                inst_caption = caption_dict[seq_name]
                # inter_relation = relation_dict[seq_name
                # out['videos'].append({'id': video_cnt, 'file_name': seq_name,\
                #                         'summary': video_summary, 'caption': inst_caption, 'relation': inter_relation})
                if split == 'test':
                    seqmap.write(seq_name + '\n')
                

                jpg_files = os.listdir(os.path.join(seq_dir_path, 'imgs'))
                jpg_files.sort()

                first_img = jpg_files[0]
                first_img_path = os.path.join(seq_dir_path, 'imgs', first_img)
                img = Image.open(first_img_path)
                img_h, img_w = img.height, img.width
                img.close()

                vid_info = {
                    "file_names": [seq_name + '/imgs/' + jpg_file for jpg_file in jpg_files],
                    "height": img_h,
                    "width": img_w,
                    "length": len(jpg_files),
                    "fps": 30,
                    "id": video_cnt
                }
                out['videos'].append(vid_info)
                

                id2anns = {}

                lbl_data = np.loadtxt(os.path.join(seq_dir_path, 'gt', 'gt.txt'), dtype=str, delimiter=',')
                for i in range(len(jpg_files)):
                    jpg_file = jpg_files[i]
                    image_cnt += 1
                    img_path = seq_name + '/imgs/' + jpg_file
                    img = Image.open(os.path.join(seq_dir_path, 'imgs', jpg_file))
                    img_h, img_w = img.height, img.width
                    img.close()
                    
                    # image_info = {
                    #     'file_names': img_path,
                    #     'id': image_cnt,
                    #     'frame_id': i + 1,
                    #     'prev_image_id': image_cnt - 1 if i > 0 else -1,
                    #     'next_image_id': image_cnt + 1 if i != len(jpg_files) - 1 else -1,
                    #     'video_id': video_cnt,
                    #     'height': img_h,
                    #     'width': img_w
                    # }
                    # out['images'].append(image_info)

                    frame_id = int(jpg_file.split('.')[0])
                    for row in lbl_data:

                        if int(row[0]) != frame_id:
                            continue
                        else:
                            track_id = int(row[1])
                            bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                            if track_id in id2anns:
                                id2anns[track_id]['bbox'][i]=bbox
                                id2anns[track_id]['areas'][i]=bbox[2] * bbox[3]
                            else :
                                # print("inst_caption:", inst_caption)
                                ann_cnt += 1
                                ann_info = {
                                    'id': ann_cnt,
                                    'category_id': 1,
                                    'video_id': video_cnt,
                                    'height': img_h,
                                    'width': img_w,
                                    'length': len(jpg_files),
                                    'bbox': [None] * len(jpg_files),
                                    'conf': 1.0,
                                    'iscrowd': 0,
                                    'areas': [None] * len(jpg_files),
                                    'caption': inst_caption[str(track_id)] if str(track_id) in inst_caption else '',
                                }
                                ann_info['bbox'][i] = bbox
                                ann_info['areas'][i] = bbox[2] * bbox[3]
                                id2anns[track_id] = ann_info
                            # out['annotations'].append(ann_info)
                
                for track_id, ann_info in id2anns.items():
                    out['annotations'].append(ann_info)
                video_cnt += 1

        print("Found {} videos, {} images, {} annotations in {} split.".format(video_cnt, image_cnt, ann_cnt, split))
        out_path = os.path.join(OUT_PATH, split + '_instances_.json')
        with open(out_path, 'w') as f:
            json.dump(out, f)      
        print("anns saved to {}".format(out_path))  
    seqmap.close()
