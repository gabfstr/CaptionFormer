# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import os
import json
import cv2
import tqdm
from pycocotools import mask as pymask
import numpy as np
import argparse

# Function to get the center and bounding box of a mask
def get_center(mask):
    h1, h2 = np.argwhere(mask.sum(axis=1).reshape(-1)).min(), np.argwhere(mask.sum(axis=1).reshape(-1)).max()
    w1, w2 = np.argwhere(mask.sum(axis=0).reshape(-1)).min(), np.argwhere(mask.sum(axis=0).reshape(-1)).max()
    return int((h1 + h2) / 2), int((w1 + w2) / 2), h1, w1, h2, w2
def get_bbox_center(box):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    return int((x1 + x2) / 2), int((y1 + y2) / 2), y1, x1, y2, x2

# Color map for visualization
# color_map = [[20, 255, 20], [20, 20, 255], [255, 20, 20], [20, 255, 255], [255, 20, 255], [255, 255, 20],
#              [42, 42, 128], [165, 42, 42], [134, 134, 103], [0, 0, 142], [255, 109, 65],
#              [0, 226, 252], [5, 121, 0], [0, 60, 100], [250, 170, 30], [100, 170, 30], [179, 0, 194],
#              [255, 77, 255], [120, 166, 157], [73, 77, 174], [0, 80, 100], [182, 182, 255], [0, 143, 149],
#              [174, 57, 255], [0, 0, 230], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151],
#              [188, 208, 182], [145, 148, 174], [106, 0, 228], [0, 0, 70], [199, 100, 0], [166, 196, 102],
#              [110, 76, 0], [133, 129, 255], [0, 0, 192], [183, 130, 88], [130, 114, 135], [107, 142, 35],
#              [0, 228, 0], [174, 255, 243], [255, 208, 186]]
# Color map for visualization
# Color map for visualization
# Color map for visualization (BGR format)
color_map = [
    [120, 240, 120],  # light green
    [255, 80, 80],    # vivid blue
    [100, 100, 255],  # coral red
    # [128, 128, 255],  # soft red
    # [255, 100, 200],  # purple
    [255, 220, 60],   # cyan
    [160, 240, 255],  # pale yellow
    [180, 170, 255],  # rose
    [180, 255, 150],  # mint green
    [255, 100, 100],  # soft indigo
    [255, 150, 100],  # periwinkle
    [200, 255, 70],   # aquamarine
    [200, 255, 128],  # teal green (later in list)
    [100, 200, 255],  # orange
    [255, 120, 220],  # magenta
    [255, 200, 128],  # baby blue
    [120, 255, 180],  # spring green
    [255, 140, 140],  # soft violet
    [230, 255, 170],  # turquoise
    [110, 210, 255],  # goldenrod
    [170, 140, 255],  # soft cherry
    [255, 180, 70],   # sky blue
    [210, 210, 210],  # gray (for fallback)
    [255, 130, 230],  # orchid
    [255, 180, 200],  # lavender
    [120, 180, 255],  # light tangerine
    [255, 200, 170],  # mist blue
    [150, 250, 170],  # green apple
    [220, 200, 255],  # light pink
    [255, 255, 140],  # pale aqua
]
contour_color = (220, 220, 220)


def draw_contours(
    frame: np.ndarray,
    contours: tuple,
    indices: int = -1, 
    thickness: int = 1, 
    color: tuple = (0, 0, 255),
    alpha: float = 0.4,
):
    """
    Draw translucent contours on a frame.

    Args:
        frame (np.ndarray): The input frame (H, W, 3).
        contours (tuple): The contours defining the mask.
        indices (int): The index of the contours to be drawn.
            Pass ``-1`` to consider all of them.
        thickness (int): The thickness of the contours.
        color (tuple): BGR color used to draw the contours.
        alpha (float): A value between 0.0 (fully transparent)
            and 1.0 (fully opaque) for the contour.
    """
    # Make a copy of the frame to draw the contours
    overlay = frame.copy()
    # Draw the contours on the overlay
    cv2.drawContours(overlay, contours, indices, color, thickness)
    # Blend the overlay with the original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# Add caption to an image frame
def add_caption_to_frame(frame, caption):
    frame_size = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_color = (255, 255, 255)
    bg_color = (150, 150, 150)
    thickness = 2
    x, y = 8, frame_size[0] - 10
    line_height = int(25 * font_scale)
    max_width = frame_size[1] - 10

    wrapped_caption = []
    words = caption.split(' ')
    current_line = ""

    for word in words:
        line_with_word = current_line + (' ' if current_line else '') + word
        text_size, _ = cv2.getTextSize(line_with_word, font, font_scale, thickness)
        if text_size[0] <= max_width:
            current_line = line_with_word
        else:
            wrapped_caption.append(current_line)
            current_line = word

    if current_line:
        wrapped_caption.append(current_line)

    total_text_height = len(wrapped_caption) * line_height
    y = frame_size[0] - 10 - total_text_height + line_height

    if y < 0:
        y = 10

    overlay = frame.copy()
    alpha = 0.6

    for i, line in enumerate(wrapped_caption):
        y_position = y + i * line_height
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_width, text_height = text_size
        cv2.rectangle(overlay, (x, y_position - text_height), (x + text_width, y_position + 5), bg_color, -1)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for i, line in enumerate(wrapped_caption):
        y_position = y + i * line_height
        frame = cv2.putText(frame, line, (x, y_position), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output/lvvis_vis/', help='Path to the output directory.')
    parser.add_argument('--gt_json', default='./datasets/LVVIS/lvviscap_val_instances.json', help='Path to the annotations JSON file.')
    parser.add_argument('--dt_json', default='./outputs/CaptionFormer/eval_ckpt/inference/results.json', help='Path to the detections JSON file.')
    parser.add_argument('--img_dir', default='./datasets/LVVIS/val/JPEGImages', help='Path to the images directory.')
    parser.add_argument('--nb_videos', type=int, default=20, help='Number of videos to process.')
    parser.add_argument('-vidstg', action='store_true', help='Visualize VidSTG format.')
    parser.add_argument('-vln', action='store_true', help='Visualize VLN format.')
    parser.add_argument('-vidstg_msk', action='store_true', help='Visualize VidSTG format with masks.')
    parser.add_argument('-caption', action='store_true', help='Visualize caption.')
    parser.add_argument('-no_cat', action='store_true', help='Do not visualize category.')
    parser.add_argument('--skip', type=int, default=0, help='Number of vids to skip.')
    parser.add_argument('-no_gt', action='store_true', help='Do not visualize GT.')
    parser.add_argument('--thresh', type=float, default=0.5, help='Threshold for visualization.')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha value for visualization.')
    parser.add_argument('-random', action='store_true', help='Randomly select videos.')
    args = parser.parse_args()
    # output_dir = 'output/lvvis_vis/'
    # anno_json = './datasets/LVVIS/val/val_instances_.json'
    # dt_json = './outputs/CaptionFormer/eval_ckpt/inference/results.json'
    # img_dir = './datasets/LVVIS/val/JPEGImages'

    if args.vidstg_msk : 
        args.vidstg = True

    dt = json.load(open(args.dt_json, 'r'))
    data = json.load(open(args.gt_json, 'r'))
    categories = data['categories']
    print("Number of categories:", len(categories))
    print("categories:", categories)
    # videos = data['videos']
    imgs = data['images']
    video_ids = set([v['video_id'] for v in imgs])
    # ###temp
    # dt_ids = [d['video_id'] for d in dt]
    # videos = [v for v in videos if v['id'] in dt_ids]
    # print("\nTemporary : test dt sample")
    # ####

    if args.random==True:
        print("Randomly selecting {} videos".format(args.nb_videos))
        video_ids = np.random.choice(list(video_ids), size=args.nb_videos, replace=False)
    else :
        nb_videos = args.nb_videos
        print("Taking the first {} to test".format(nb_videos))
        # videos = videos[:nb_videos]
        video_ids = list(video_ids)[:nb_videos]
    print("Number of videos:", len(video_ids))


    ######### TEMPORARY
    # video_ids = video_ids[50:]
    # print("selected the last {} videos".format(len(video_ids)))
    ######
    # shortlist = [26, 28, 30, 36, 41, 43]
    # print("TEMPORARY : shortlist : ", shortlist)
    # video_ids = [v for v in video_ids if v in shortlist]
    # print("video_ids: ", video_ids)
    # raise ValueError("Temporary stop here")
    ######### TEMPORARY


    # Create directories for GT and DT outputs
    os.makedirs(args.output_dir, exist_ok=True)
    # outputdir = os.path.join(output_dir, 'dt')
    # output_gt_dir = os.path.join(output_dir, 'gt')
    # os.makedirs(output_dt_dir, exist_ok=True)
    # os.makedirs(output_gt_dir, exist_ok=True)

    # Build dictionaries for category names and detections
    dt_dic = {}
    category_dic = {}
    for category in categories:
        category_dic[category['id']] = category['name']

    # for d in dt:
    #     if d['video_id'] not in dt_dic.keys():
    #         dt_dic[d['video_id']] = []
    #     dt_dic[d['video_id']].append(d)

    to_skip = args.skip
    # Process each video
    for video_id in tqdm.tqdm(video_ids):
        if to_skip > 0:
            to_skip -= 1
            print("skipping video ", video_id)
            continue
        img_infos = [img for img in imgs if img['video_id'] == video_id]
        #sort by id
        img_infos.sort(key=lambda x: x['id'])
        test = [d for d in dt if d['image_id'] == 6000]
        
        # print("video : ", video)
        if args.vidstg :
            if "file_name" in img_infos[0]:
                # print("img_infos[0] : ", img_infos[0])
                video_full_name = os.path.dirname(img_infos[0]['file_name'])
                # print("video_name : ", video_full_name)
                try :
                    video_prefix, video_name = video_full_name.split('/')[0], video_full_name.split('/')[1]
                except :
                    video_prefix=''
                    video_name = video_full_name
                img_list = [img['file_name'] for img in img_infos]
                video_folder = os.path.join(args.output_dir, video_full_name)
            else :
                print("img_infos : ", img_infos)
                raise ValueError("img_infos[0] does not have file_name")
                video_name = video['file_name'].split('.mp4')[0]
                video_length = video['length']
                print("video_length : ", video_length)
                img_list = os.listdir(os.path.join(args.img_dir, video_name))
                print("len(img_list) : ", len(img_list))
                img_list = img_list[:video_length]
                print("len(img_list) : ", len(img_list))
                img_list = [os.path.join(video_name, img) for img in img_list]
                video_folder = os.path.join(args.output_dir, video_name.split('/')[0])
                video_name = video_name.split('/')[1]
        else :
            video_name = img_infos[0]['file_name'].split('/')[0]
            img_list = [img['file_name'] for img in img_infos]
            video_folder = os.path.join(args.output_dir, video_name)

        img_list.sort()
        # video_dt = dt_dic.get(video_id, [])

        # # Sort them by score (highest first)
        # video_dt.sort(key=lambda x: x['score'], reverse=True)

        # Initialize video writers
        first_img_path = os.path.join(args.img_dir, img_list[0])
        first_img = cv2.imread(first_img_path)
        height, width, _ = first_img.shape

        video_len = len(img_list)
        # sample 4 vid ids uniformly
        cap_sampled_ids = [i for i in range(0, video_len, (video_len//4)+1)]


        os.makedirs(video_folder, exist_ok=True)
        dt_video_path = os.path.join(video_folder, f"{video_name}_dt.mp4")
        gt_video_path = os.path.join(video_folder, f"{video_name}_gt.mp4")
        # print("dt_video_path : ", dt_video_path)
        # print("gt_video_path : ", gt_video_path)
        # print("width : ", width)
        # print("height : ", height)
        dt_writer = cv2.VideoWriter(dt_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        if not args.no_gt:
            gt_writer = cv2.VideoWriter(gt_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))      
        
        track_id_dict = {}
        track_id_counter = 0

        for fid, img_info in enumerate(img_infos):
            # fid = img_info['id']
            img_path = img_info['file_name']
        # for fid, img_path in enumerate(img_list):
            img = cv2.imread(os.path.join(args.img_dir, img_path))
            img_dt = img.copy()
            h, w, _ = img.shape

            img_pred = [d for d in dt if d['image_id'] == img_info['id']]
            # print("frame {} found {} predictions".format(fid, len(img_pred)))
            num_cap = 0
            # Visualization for detections (DT)
            mask_vis = np.zeros((h, w, 3))
            for obj in img_pred:
                obj_id = obj['track_id']
                img_dt_obj = img_dt.copy()
                category_id = obj['category_id']
                # print("category_id", category_id)
                # if not args.vidstg_msk:
                category_name = category_dic[category_id]
                # else : 
                # category_name = ''
                # print("category_name", category_name)
                score = obj['score']
                # print("score", score)
                # thresh=0.5
                thresh=args.thresh
                if score < thresh:
                    # print("skipping")
                    continue
                if obj_id not in track_id_dict.keys():
                    track_id_dict[obj_id] = track_id_counter
                    track_id_counter += 1
                obj_id = track_id_dict[obj_id]
                # print("\nkept")
                if args.vidstg and not args.vidstg_msk :
                    # print("len(obj['bbox']) : ", len(obj['bbox']))
                    # print("fid : ", fid)
                    try :
                        bbox = obj['bbox']
                    except IndexError :
                        print("IndexError")
                        print("fid : ", fid)
                        print("obj : ", len(obj))
                        raise ValueError
                        continue
                    if bbox == None:
                        continue
                    # x1, y1, x2, y2 = bbox
                    x1, y1, w_, h_ = bbox
                    x2, y2 = x1 + w_, y1 + h_
                    color = color_map[int(obj_id) % len(color_map)]
                    img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    h_, w_, y1, x1, y2, x2 = get_bbox_center(bbox)
                    if args.caption:
                        img_dt_obj = cv2.rectangle(img_dt_obj, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                else :
                    obj_mask = pymask.decode(obj['segmentation'])
                    if obj_mask.sum() == 0:
                        continue
                    color = color_map[int(obj_id) % len(color_map)]
                    mask_vis[obj_mask > 0] = color
                    img[obj_mask > 0] = img[obj_mask > 0] * (1-args.alpha) + mask_vis[obj_mask > 0] * args.alpha
                    contours, _ = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # draw contours with the color of the mask 
                    # img = cv2.drawContours(img, contours, -1, (222, 222, 222), 2)
                    # img = cv2.drawContours(img, contours, -1, color, 2)
                    img =  draw_contours(img, contours, -1, 2, contour_color, 0.4)
                    h_, w_, y1, x1, y2, x2 = get_center(obj_mask)
                    if args.caption:
                        img_dt_obj[obj_mask > 0] = img_dt_obj[obj_mask > 0] * (1-args.alpha) + mask_vis[obj_mask > 0] * args.alpha
                        # img_dt_obj = cv2.drawContours(img_dt_obj, contours, -1, color, 2)
                        img_dt_obj = draw_contours(img_dt_obj, contours, -1, 2, contour_color, 0.4)
                if args.no_cat is not True:
                    cat_name_score = f"{category_name}:{score:.2f}"
                    img = cv2.putText(img, cat_name_score, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
                    img = cv2.putText(img, cat_name_score, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
                # #Add score
                # img_dt = cv2.putText(img_dt, f"{score:.2f}", ((x1 + x2) // 2 - 45, (y1 + y2) // 2 + 25),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
                # img_dt = cv2.putText(img_dt, f"{score:.2f}", ((x1 + x2) // 2 - 45, (y1 + y2) // 2 + 25),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
            
                if args.caption and num_cap < 10 and fid in cap_sampled_ids :
                    num_cap += 1
                    if args.no_cat is not True:
                        img_dt_obj = cv2.putText(img_dt_obj, cat_name_score, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
                        img_dt_obj = cv2.putText(img_dt_obj, cat_name_score, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
                    caption = obj['caption']
                    if isinstance(caption, list):
                        caption = caption[0]
                    if caption is None :
                        caption = ''
                    # print("caption", caption)
                    # dt_img_path = dt_video_path.replace("_dt.mp4", f"_dt_{num_cap}.jpg")
                    dt_img_path = os.path.join(video_folder, f"cap_{obj_id}",f"frame_{fid:04d}.jpg")
                    os.makedirs(os.path.dirname(dt_img_path), exist_ok=True)
                    img_dt_obj = add_caption_to_frame(img_dt_obj, caption)
                    cv2.imwrite(dt_img_path, img_dt_obj)

            dt_writer.write(img)


            if args.no_gt:
                continue

            # vid_ann = [ann for ann in data['annotations'] if ann['video_id'] == video['id']]
            img_ann = [ann for ann in data['annotations'] if ann['image_id'] == img_info['id']]

            # Visualization for ground truth (GT)
            gt_img = cv2.imread(os.path.join(args.img_dir, img_path))
            gt_img_dt = gt_img.copy()
            mask_vis = np.zeros((h, w, 3))
            for annotation in img_ann:
                obj_id = annotation['track_id']
                gt_img_obj = gt_img_dt.copy()
        
                category_id = annotation['category_id']
                # print("gt category_id", category_id)
                category_name = category_dic[category_id]
                # print("gt category_name", category_name)
                if args.vidstg :
                    try :
                        bbox = annotation['bbox']
                    except IndexError :
                        print("IndexError")
                        continue
                    if bbox == None:
                        continue
                    # print("gt bbox", bbox)
                    x1, y1, w_, h_ = bbox
                    x2, y2 = x1 + w_, y1 + h_
                    color = color_map[int(obj_id) % len(color_map)]
                    gt_img = cv2.rectangle(gt_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    gt_img = cv2.putText(gt_img, category_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    if args.caption:
                        gt_img_obj = cv2.rectangle(gt_img_obj, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        gt_img_obj = cv2.putText(gt_img_obj, category_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else :
                    segm = annotation['segmentation']
                    if segm == None:
                        continue
                    obj_mask = pymask.decode(segm)
                    # color = np.array(color_map[annotation['category_id'] % len(color_map)])  # Convert to NumPy array
                    color = color_map[int(obj_id) % len(color_map)]
                    mask_vis[obj_mask > 0] = color
                    gt_img[obj_mask > 0] = gt_img[obj_mask > 0] * (1-args.alpha) + mask_vis[obj_mask > 0] * args.alpha
                    
                    contours, _ = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # gt_img = cv2.drawContours(gt_img, contours, -1, color, 2)
                    gt_img = draw_contours(gt_img, contours, -1, 2, color, 0.4)
                    h_, w_, y1, x1, y2, x2 = get_center(obj_mask)
                    if args.caption:
                        gt_img_obj[obj_mask > 0] = gt_img_obj[obj_mask > 0] * (1-args.alpha) + mask_vis[obj_mask > 0] * args.alpha
                        # gt_img_obj = cv2.drawContours(gt_img_obj, contours, -1, color, 2)
                        gt_img_obj = draw_contours(gt_img_obj, contours, -1, 2, color, 0.4)
                if args.no_cat is not True:
                    gt_img = cv2.putText(gt_img, category_name, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
                    gt_img = cv2.putText(gt_img, category_name, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
                if args.caption and obj_id < 10 and fid in cap_sampled_ids :
                    if args.no_cat is not True:
                        gt_img_obj = cv2.putText(gt_img_obj, category_name, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
                        gt_img_obj = cv2.putText(gt_img_obj, category_name, ((x1 + x2) // 2 - 45, (y1 + y2) // 2 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
                    caption = annotation['caption']
                    if isinstance(caption, list):
                        caption = caption[0]
                    if caption is None :
                        caption = ''
                    gt_img_obj_path = gt_video_path.replace("_gt.mp4", f"_gt_{obj_id}.jpg")
                    gt_img_obj = add_caption_to_frame(gt_img_obj, caption)
                    cv2.imwrite(gt_img_obj_path, gt_img_obj)

            gt_writer.write(gt_img)

        dt_writer.release()
        if not args.no_gt:
            gt_writer.release()

    print("Processing complete. Videos saved.")
