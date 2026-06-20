# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import os
import json
from collections import defaultdict

# Define paths for input files and output
split = "train"
# split = "val"

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


annotation_folder_path = "./datasets/VidSTG/annotations/training/"

if split == "train" :
    annotation_file_path = "./datasets/VidSTG/annotations/train_annotations.json"
    output_file_path = "./datasets/VidSTG/annotations/train_instances_.json"
elif split == "val" :
    annotation_file_path = "./datasets/VidSTG/annotations/val_annotations.json"
    output_file_path = "./datasets/VidSTG/annotations/val_instances_.json"
else :
    raise Exception("split must be in ['train', 'val']")

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def create_coco_annotations(annotation_file_path, annotation_folder_path):
    # Load the main annotation file
    main_annotations = load_json(annotation_file_path)
    
    # vid_list = list(set([vid["vid"] for vid in main_annotations]))
    # print("in main annotations, found ", len(vid_list), "unqiue videos")
    vid_list_with_caption = list(set([vid["vid"] for vid in main_annotations if vid["captions"]]))
    print("in main annotations, found ", len(vid_list_with_caption), "unqiue videos with captions")
    # raise ValueError("Stop")

    # print("vid_list_with_caption", vid_list_with_caption)
    # print(" 359808038 in vid lis ? ", "359808038" in vid_list_with_caption)
    # print(" 359808038 in vid lis ? ", 359808038 in vid_list_with_caption)
    # raise ValueError("Stop")
    # Initialize the output structure
    coco_annotations = {
        "videos": [],
        "categories": VIDSTG_CATEGORIES,
        "annotations": []
    }

    category_map = {a['name']:a['id'] for a in VIDSTG_CATEGORIES}

    # Create a video ID map and counters
    video_id_map = {}
    video_id_counter = 0

    # Annotation ID counter
    annotation_id_counter = 1

    # Map video captions for quick lookup
    video_ids = [video["vid"] for video in main_annotations]
    # print("video_ids", video_ids)
    # print("vid_list_with_caption", vid_list_with_caption)
    video_captions = {
        vid : [] for vid in video_ids
        #video["vid"]: video["captions"] for video in main_annotations
    }

    video_used_segments = {}

    for video in main_annotations:
        video_captions[video["vid"]].append(video["captions"])
        if video["vid"] not in video_used_segments:
            video_used_segments[video["vid"]] = video["used_segment"]
        else :
            print("vid :", video["vid"])
            print("new used segment", video["used_segment"])
            print("old used segment", video_used_segments[video["vid"]])

            assert video_used_segments[video["vid"]] == video["used_segment"]; "used_segment must be the same for all annotations of a video"

    # Iterate over shard folders
    for shard_folder in sorted(os.listdir(annotation_folder_path)):
        shard_path = os.path.join(annotation_folder_path, shard_folder)
        if not os.path.isdir(shard_path):
            continue

        # Iterate over video JSON files in the shard folder
        for video_file in sorted(os.listdir(shard_path)):
            if not video_file.endswith(".json"):
                continue

            print("Processing", video_file)
            video_path = os.path.join(shard_path, video_file)
            detailed_annotations = load_json(video_path)

            video_name = detailed_annotations["video_id"]
            # print("video_name", video_name)
            # # Str or int ? 
            # print("video_name is str :", isinstance(video_name, str))
            # print("video_name is int :", isinstance(video_name, int))
            # if video_name =="7290684468":
            #     raise ValueError("Stop")
            video_path = detailed_annotations["video_path"]
            video_height = detailed_annotations["height"]
            video_width = detailed_annotations["width"]
            video_frame_count = detailed_annotations["frame_count"]
            video_fps = detailed_annotations["fps"]

            used_segment = video_used_segments[video_name]

            #Check that the vid is present in the main annotations
            if video_name not in vid_list_with_caption:
                print("Video not in main annotations, skipping")
                continue

            # Map the video to a new ID
            video_id_map[video_name] = video_id_counter
            video_id = video_id_counter
            video_id_counter += 1

            # Add video entry to the output
            coco_annotations["videos"].append({
                "height": video_height,
                "width": video_width,
                "fps": video_fps,
                "length": video_frame_count,
                "used_segment": used_segment,
                "file_name": video_path,
                "id": video_id
            })

            # # Process categories and add them to the category map
            # for obj in detailed_annotations["subject/objects"]:
            #     category_name = obj["category"]
            #     if category_name not in category_map:
            #         category_map[category_name] = category_id_counter
            #         coco_annotations["categories"].append({
            #             "id": category_id_counter,
            #             "name": category_name,
            #             "partition": 1
            #         })
            #         category_id_counter += 1

            # Process each frame and its annotations
            trajectories = detailed_annotations["trajectories"]

            # Collect annotations by object ID
            object_annotations = defaultdict(lambda: {
                "tid": None,
                "bbox": [None] * video_frame_count,
                "areas": [None] * video_frame_count,
                "length": 0,
                "height": video_height,
                "width": video_width,
                "category_id": None,
                "caption": None
            })

            for frame_idx, frame_annotations in enumerate(trajectories):
                for obj in frame_annotations:
                    tid = obj["tid"]
                    object_annotations[tid]["tid"] = tid
                    bbox = obj["bbox"]

                    # Calculate the area of the bounding box
                    area = (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"])

                    # Update the box and area for the current frame converted in COCO format
                    object_annotations[tid]["bbox"][frame_idx] = [
                        bbox["xmin"], bbox["ymin"], bbox["xmax"] - bbox["xmin"], bbox["ymax"] - bbox["ymin"]
                    ]
                    # object_annotations[tid]["bbox"][frame_idx] = [
                    #     bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
                    # ]
                    object_annotations[tid]["areas"][frame_idx] = area
                    object_annotations[tid]["length"] += 1

                    # Assign the category ID
                    if object_annotations[tid]["category_id"] is None:
                        object_annotations[tid]["category_id"] = category_map[detailed_annotations["subject/objects"][tid]["category"]]

            # Add captions for each annotation
            video_cap = video_captions.get(video_name, [])
            # Filter to keep only the first with the right target_id
            tgt_id_seen = set()
            for caption_entry in video_cap:

                target_id = caption_entry[0]["target_id"]
                if target_id in tgt_id_seen:
                    continue
                tgt_id_seen.add(target_id)
                if target_id in object_annotations:
                    object_annotations[target_id]["caption"] = caption_entry[0]["description"]

            # Add the annotations to the output
            for tid, obj_data in object_annotations.items():
                # print("obj data", obj_data)
                # raise ValueError("Stop")
                coco_annotations["annotations"].append({
                    "video_id": video_id,
                    "iscrowd": 0,
                    "height": obj_data["height"],
                    "width": obj_data["width"],
                    "length": obj_data["length"],
                    "bbox": obj_data["bbox"],
                    "category_id": obj_data["category_id"],
                    "id": annotation_id_counter,
                    "areas": obj_data["areas"],
                    "caption": obj_data["caption"]
                })
                annotation_id_counter += 1


    # # iterate through categories and remap id to alphabetical order
    # cats = coco_annotations['categories']
    # id_map = {}
    # cat_names = sorted([f["name"] for f in cats])
    # for i, cat in enumerate(cat_names):
    #     id_map[cat] = i+1
    # #Edit id in the data 
    # for cat in cats:
    #     cat["id"] = id_map[cat["name"]]
    # #Sort by id
    # cats = sorted(cats, key=lambda x: x['id'])
    # # print("new categories:", cats)
    # #Save the annotations with the new categories sorted by id
    # coco_annotations['categories'] = cats
    # #put it as first in keys
    # coco_annotations = {k: coco_annotations[k] for k in ['categories', 'videos', 'annotations']}
    
    print("total number of videos :", len(coco_annotations["videos"]))
    print("total number of categories :", len(coco_annotations["categories"]))
    print("total number of annotations :", len(coco_annotations["annotations"]))

    return coco_annotations

# Generate the COCO-style annotations
coco_annotations = create_coco_annotations(annotation_file_path, annotation_folder_path)

# Save to file
with open(output_file_path, "w") as f:
    json.dump(coco_annotations, f, indent=4)

print(f"Converted annotations saved to {output_file_path}")
