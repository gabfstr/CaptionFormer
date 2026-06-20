# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import os
import json
import subprocess

from argparse import ArgumentParser

# Configuration
parser = ArgumentParser(description="Convert VidSTG videos to JPG frames")
parser.add_argument('--train_only', default=True, help='Process only training set videos')
parser.add_argument('--val_only', default=False, help='Process only validation set videos')
parser.add_argument('--reverse', default=False, help='Reverse the order of processing')
parser.add_argument('--start_middle', default=False, help='Start processing from the middle folder')
args = parser.parse_args()

train_only = args.train_only
val_only = args.val_only
if val_only:
    train_only = False
    print("val only set to True, setting train_only to False")
reverse = args.reverse
start_middle = args.start_middle


# Define the root path containing the folders with .mp4 files
root_path = "./datasets/VidSTG/video/"
ann_path = "./datasets/VidSTG/annotations/train_instances_200_frames_.json"
ann_path_val = "./datasets/VidSTG/annotations/vidstg_max200f_val_instances_.json"

with open(ann_path, 'r') as f:
    ann = json.load(f)
with open(ann_path_val, 'r') as f:
    ann_val = json.load(f)

vid_data = ann['videos']
vid_data_val = ann_val['videos']
id_2_len = {a['file_name']: a['length'] for a in vid_data}
id_2_len_val = {a['file_name']: a['length'] for a in vid_data_val}



# Function to extract frames using FFmpeg
def extract_frames_with_ffmpeg(root_path):

    subdirs = sorted([os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    
    # reverse 
    if reverse:
        subdirs = subdirs[::-1]
    if start_middle :
        subdirs = subdirs[len(subdirs)//2:] 

    num_folders = len(subdirs)

    for i, subdir in enumerate(subdirs, start=1):
        # print("\n\n Processing folder {}/{}: {}".format(i, num_folders, subdir))
        files = sorted(os.listdir(subdir))  # Sort files alphabetically
        
        for _, file in enumerate(files):
            if file.endswith(".mp4"):
                video_path = os.path.join(subdir, file)
                video_name = os.path.splitext(file)[0]
                output_folder = os.path.join(subdir, video_name)
                
                ann_file_name = os.path.join(os.path.basename(subdir), file)
                # print(ann_file_name)
                if ann_file_name in id_2_len:
                    true_len = id_2_len[ann_file_name]
                else :
                    if train_only : 
                        continue 
                if ann_file_name in id_2_len_val:
                    true_len = id_2_len_val[ann_file_name]
                else :
                    if val_only :
                        print("Skipping, val_only") 
                        continue
                if ann_file_name not in id_2_len and ann_file_name not in id_2_len_val:
                    print(f"File {ann_file_name} not found in annotations.")
                    continue


                # Check if the folder already exists and skip processing if it does
                if os.path.exists(output_folder):
                    file_len = len(os.listdir(output_folder))
                    print(f"File len: {file_len}")
                    print(f"True len: {true_len}")
                    # print("output_folder: ", output_folder)
                    # print("os.listdir(output_folder): ", sorted(os.listdir(output_folder)))
                    if file_len >= true_len:
                        print(f"Skipping {video_path}: Frames already extracted.")
                        continue
                    else:
                        print(f"Reprocessing {video_path}: Frames already extracted but len mismatch.")
                

                
                # Create a folder for the frames
                os.makedirs(output_folder, exist_ok=True)
                
                # Construct FFmpeg command
                output_pattern = os.path.join(output_folder, "%04d.jpg")
                command = [
                    "ffmpeg",
                    "-vsync", "0",  # Prevents ffmpeg from duplicating or dropping frames
                    "-i", video_path,         # Input video
                    "-q:v", "2",             # Set JPEG quality (lower is better quality)
                    output_pattern           # Output frame pattern
                ]
                
                # Execute FFmpeg command
                try:
                    subprocess.run(command, check=True)
                    print(f"Processed {video_path} -> {output_folder}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {video_path}: {e}")
                

# Call the function
extract_frames_with_ffmpeg(root_path)
