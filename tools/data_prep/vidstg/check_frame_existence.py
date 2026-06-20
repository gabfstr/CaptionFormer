#!/usr/bin/env python3
# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""
Check that all frames in VidSTG annotation file exist as JPG files.

Verifies that each frame referenced in the annotation file exists on disk.

Usage:
    python check_frame_existence.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def check_frames_exist(annotation_path, video_base_path):
    """
    Check that all frames in annotation file exist as JPG files.
    
    Args:
        annotation_path: Path to annotation JSON file
        video_base_path: Base path where video frames are stored
    
    Returns:
        dict: Statistics about missing/existing frames
    """
    print(f"Loading annotations from: {annotation_path}")
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    print(f"Base video path: {video_base_path}")
    print(f"\nDataset info:")
    print(f"  - Videos: {len(data['videos'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    
    # Statistics
    total_frames = 0
    total_videos = len(data['videos'])
    missing_frames = []
    existing_frames = 0
    videos_with_missing_frames = set()
    frames_per_video = []
    
    print("\nChecking frame existence...")
    
    for video_idx, video in enumerate(data['videos']):
        video_id = video['id']
        file_names = video['file_names']
        video_length = video['length']
        
        frames_per_video.append(video_length)
        total_frames += video_length
        
        # Check each frame
        video_missing_count = 0
        for frame_idx, frame_path in enumerate(file_names):
            full_path = os.path.join(video_base_path, frame_path)
            
            if not os.path.exists(full_path):
                missing_frames.append({
                    'video_id': video_id,
                    'video_idx': video_idx,
                    'frame_idx': frame_idx,
                    'frame_path': frame_path,
                    'full_path': full_path
                })
                videos_with_missing_frames.add(video_id)
                video_missing_count += 1
            else:
                existing_frames += 1
        
        # Progress update every 100 videos
        if (video_idx + 1) % 100 == 0:
            print(f"  Processed {video_idx + 1}/{total_videos} videos...")
    
    # Compute statistics
    stats = {
        'total_videos': total_videos,
        'total_frames': total_frames,
        'existing_frames': existing_frames,
        'missing_frames_count': len(missing_frames),
        'videos_with_missing_frames': len(videos_with_missing_frames),
        'videos_complete': total_videos - len(videos_with_missing_frames),
        'min_frames_per_video': min(frames_per_video) if frames_per_video else 0,
        'max_frames_per_video': max(frames_per_video) if frames_per_video else 0,
        'avg_frames_per_video': sum(frames_per_video) / len(frames_per_video) if frames_per_video else 0,
    }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nTotal statistics:")
    print(f"  - Total videos: {stats['total_videos']}")
    print(f"  - Total frames: {stats['total_frames']}")
    print(f"  - Existing frames: {stats['existing_frames']}")
    print(f"  - Missing frames: {stats['missing_frames_count']}")
    print(f"  - Success rate: {100 * stats['existing_frames'] / stats['total_frames']:.2f}%")
    
    print(f"\nVideo statistics:")
    print(f"  - Videos with all frames: {stats['videos_complete']}")
    print(f"  - Videos with missing frames: {stats['videos_with_missing_frames']}")
    print(f"  - Complete rate: {100 * stats['videos_complete'] / stats['total_videos']:.2f}%")
    
    print(f"\nFrame count per video:")
    print(f"  - Min: {stats['min_frames_per_video']}")
    print(f"  - Max: {stats['max_frames_per_video']}")
    print(f"  - Avg: {stats['avg_frames_per_video']:.2f}")
    
    # Report missing frames
    if missing_frames:
        print(f"\n⚠️  WARNING: {len(missing_frames)} frames are missing!")
        print(f"\nFirst 20 missing frames:")
        for i, frame_info in enumerate(missing_frames[:20]):
            print(f"  {i+1}. Video {frame_info['video_id']} (idx={frame_info['video_idx']}), "
                  f"Frame {frame_info['frame_idx']}: {frame_info['frame_path']}")
        
        if len(missing_frames) > 20:
            print(f"  ... and {len(missing_frames) - 20} more")
        
        # Save detailed report
        report_path = "missing_frames_report.json"
        print(f"\nSaving detailed report to: {report_path}")
        with open(report_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'missing_frames': missing_frames
            }, f, indent=2)
    else:
        print("\n✓ SUCCESS: All frames exist!")
    
    return stats


def main():
    annotation_path = "./datasets/VidSTG/annotations/vidstg_max200f_val_instances_.json"
    video_base_path = "./datasets/VidSTG/video"
    
    # Check if paths exist
    if not os.path.exists(annotation_path):
        print(f"ERROR: Annotation file not found: {annotation_path}")
        return
    
    if not os.path.exists(video_base_path):
        print(f"ERROR: Video base path not found: {video_base_path}")
        return
    
    stats = check_frames_exist(annotation_path, video_base_path)
    
    return stats


if __name__ == "__main__":
    main()
