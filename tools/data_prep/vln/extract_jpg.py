# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

"""Extract sparse + offset jpg frames from UVO videos for VLN.

Reads the official UVO sparse video annotation (`UVO_sparse_{split}_video.json`,
which has `videos[].file_names` listing the sparse-annotated frame paths) and
extracts those frames PLUS the offset frames `{-25, -20, -15, -10, -5, 0}`
needed by the model at inference time on the extended/dense annotation.

No dependency on any homemade merged annotation — direct from UVO.
"""
import argparse
import json
import os

import cv2
from tqdm import tqdm


OFFSETS = (-25, -20, -15, -10, -5, 0)


def split_single_video(video_path, frames_dir, idxs):
    """Extract the given frame indices from a single mp4 as PNGs."""
    cap = cv2.VideoCapture(video_path)
    idxs_set = set(idxs)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cnt in idxs_set:
            success, buffer = cv2.imencode(".png", frame)
            if success:
                out_path = os.path.join(frames_dir, f"{cnt}.png")
                with open(out_path, "wb") as f:
                    f.write(buffer.tobytes())
                    f.flush()
        cnt += 1
    cap.release()
    return cnt  # total frames in the video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uvo_ann', required=True,
                    help='Path to UVO_sparse_{split}_video.json (official UVO file).')
    ap.add_argument('--video_dir', required=True, help='Directory containing UVO mp4 files.')
    ap.add_argument('--frames_dir', required=True, help='Output directory for extracted jpg frames.')
    ap.add_argument('--no_offsets', action='store_true',
                    help='Extract only sparse frames (skip the inference-time offset frames).')
    args = ap.parse_args()

    with open(args.uvo_ann, 'r') as f:
        uvo = json.load(f)

    os.makedirs(args.frames_dir, exist_ok=True)
    print(f"Extracting frames for {len(uvo['videos'])} videos → {args.frames_dir}")

    for vid in tqdm(uvo['videos']):
        # UVO file_names are like "<ytid>/<frame_idx>.png"
        file_names = vid['file_names']
        vid_id = os.path.dirname(file_names[0])
        sparse_idxs = [int(os.path.splitext(os.path.basename(p))[0]) for p in file_names]

        video_path = os.path.join(args.video_dir, vid_id + '.mp4')
        v_frame_dir = os.path.join(args.frames_dir, vid_id)
        os.makedirs(v_frame_dir, exist_ok=True)

        # First pass to learn the video length; cheap if we early-exit at max(needed_idxs).
        # Simpler: pass a generous superset; OpenCV will just skip past last frame.
        if args.no_offsets:
            idxs = sparse_idxs
        else:
            # We don't know video length yet — use a large clip and let split_single_video
            # filter; then clip negatives away.
            idxs = sorted({t + off for t in sparse_idxs for off in OFFSETS if t + off >= 0})

        split_single_video(video_path, v_frame_dir, idxs)

    print(f"Done. Extracted frames for {len(uvo['videos'])} videos.")


if __name__ == "__main__":
    main()
