# Data Preparation

We provide instructions to prepare the data to train and evaluate CaptionFormer. This includes raw images/videos downloads, and commands to build the annotations from scratch. We also provide pre-processed annotation files: [[LVIScap & LV-VIScap](https://drive.google.com/drive/folders/1qFNx7AWeXmjJ8aGiFqeO3SMoOCe6ga0D?usp=sharing)] [[Benchmarks](https://drive.google.com/drive/folders/11E86QHJW7tJw8AvlZAUEdsWs_A3DpJ7l?usp=sharing)].


All datasets are expected under `$DETECTRON2_DATASETS/` (default: `./datasets/`).
Pre-extracted CLIP features are required per dataset (see [Pre-extracted features](#clip-image-features)).

**Contents**
- [LVIScap & LV-VIScap](#lviscap--lvviscap) — released alongside the code
  - [LVIScap (image)](#lviscap-image)
  - [LV-VIScap (video)](#lvviscap-video)
- [Benchmarks](#benchmarks) — downstream evaluation
  - [VidSTG](#vidstg)
  - [VLN](#vln)
  - [BenSMOT](#bensmot)
- [Pre-extracted features](#pre-extracted-features)
  - [CLIP image features](#clip-image-features)
  - [BLIP-2 features (optional training speedup)](#blip-2-features-training-only)

---

## LVIScap & LV-VIScap

LVIScap and LV-VIScap are released alongside CaptionFormer. [[Google Drive](https://drive.google.com/drive/folders/1qFNx7AWeXmjJ8aGiFqeO3SMoOCe6ga0D?usp=sharing)] 

### LVIScap (image)

LVIScap is based on [LVIS](https://www.lvisdataset.org/) images and annotations, extended with our synthetic instance-level captions.

**Download:**
- Images: `train2017/` and `val2017/` from [COCO](https://cocodataset.org/#download).
- LVIScap annotations (`lviscap_v1_train.json`, `lviscap_v1_val.json`): from our [Google Drive folder](https://drive.google.com/drive/folders/1qFNx7AWeXmjJ8aGiFqeO3SMoOCe6ga0D?usp=share_link).

**Expected layout:**
```
datasets/
  coco/{train2017,val2017}/                       # images (shared with LVIS)
  lvis/lviscap_v1_train.json
  lvis/lviscap_v1_val.json
```

<details>
<summary><b>LVIScap + COCO merged training annotation</b> — optional training file</summary>

Download `instances_train2017.json` from [COCO](https://cocodataset.org/#download) and place it under `datasets/coco/annotations/`.

Updated layout:
```
datasets/
  coco/{train2017,val2017}/
  coco/annotations/instances_train2017.json     # added
  lvis/lviscap_v1_train.json
  lvis/lviscap_v1_val.json
```

Run the merge script:
```bash
python tools/data_prep/lviscap_lvviscap/create_coco_plus_lvis.py
# writes datasets/lvis/lviscap_v1_train+coco.json
```

</details>

---

### LV-VIScap (video)

LV-VIScap is based on [LV-VIS](https://github.com/haochenheheda/LVVIS) videos and annotations, extended with our synthetic track-level captions.

**Download:**
- Training and validation videos from [LV-VIS](https://github.com/haochenheheda/LVVIS).
- LV-VIScap annotations (`lvviscap_train_instances.json`, `lvviscap_val_instances.json`): from our [Google Drive folder](https://drive.google.com/drive/folders/1qFNx7AWeXmjJ8aGiFqeO3SMoOCe6ga0D?usp=share_link).

**Expected layout:**
```
datasets/
  LVVIS/
    train/JPEGImages/
    val/JPEGImages/
    lvviscap_train_instances.json
    lvviscap_val_instances.json
```

**Build the COCO format annotation file for evaluation:**
```bash
python tools/data_prep/lvvis2coco_annotations.py \
    --gt_json datasets/LVVIS/lvviscap_val_instances.json
# writes lvviscap_val_instances_coco_format.json next to the input
```

---

## Benchmarks
VidSTG, VLN, BenSMOT are used for donwstream evaluation.
We provide preprocessed annotation file for evaluation and training. [[Google Drive](https://drive.google.com/drive/folders/11E86QHJW7tJw8AvlZAUEdsWs_A3DpJ7l?usp=sharing)].
<!-- To recreate the VidSTG and VLN annotations from scratch, please follow instruction from [DVOC-DS](https://github.com/google-research/scenic/tree/main/scenic/projects/densevoc). -->


### VidSTG

Since CaptionFormer can be trained using category labels, we include them in the training annotations. For evaluation we use the same annotations as [DVOC-DS](https://github.com/google-research/scenic/tree/main/scenic/projects/densevoc). All are available at this [Google Drive folder](https://drive.google.com/drive/folders/11E86QHJW7tJw8AvlZAUEdsWs_A3DpJ7l?usp=sharing).

**Download:**
- VidSTG videos (`.mp4`) from [VidSTG-Dataset](https://github.com/Guaranteer/VidSTG-Dataset).
- Pre-processed VidSTG annotations from our [Google Drive folder](https://drive.google.com/drive/folders/11E86QHJW7tJw8AvlZAUEdsWs_A3DpJ7l?usp=sharing).

**Expected raw layout:**
```
datasets/VidSTG/
  video/<video_id>.mp4
  annotations/
    train_instances_200_frames_.json
    vidstg_max200f_val_instances_.json
```

**Step 1 — Extract jpg frames from mp4** (requires ffmpeg):
```bash
python tools/data_prep/vidstg/convert_vidstg_to_jpg.py --val_only True
# Same with --train_only True for the training set.
```

**Step 2 — Build the COCO format annotation file for evaluation:**
```bash
python tools/data_prep/lvvis2coco_annotations.py \
    --gt_json datasets/VidSTG/annotations/vidstg_max200f_val_instances_.json \
    --out_path datasets/VidSTG/annotations/vidstg_max200f_val_coco_format.json
```

**Final expected layout:**
```
datasets/VidSTG/
  video/<video_id>/<NNNN>.jpg                             # extracted frames
  annotations/
    train_instances_200_frames_.json                      # model input — train
    vidstg_max200f_val_instances_.json                    # model input — val
    vidstg_max200f_val_coco_format.json                   # coco format eval annotations
```

<details>
<summary><b>Full annotation reproduction</b> — rebuild the JSONs from raw VidSTG (optional)</summary>

Additional downloads:
- Videos and annotations from [VidOR](https://xdshang.github.io/docs/vidor.html) 
- Annotations from [VidSTG-Dataset](https://github.com/Guaranteer/VidSTG-Dataset)

**Raw input layout:**
```
datasets/VidSTG/
  video/<video_id>.mp4                                  # videos
  annotations/
    val_annotations.json                                # raw VidSTG val
    training.json                                       # VidOR training annotation
    validation.json                                     # VidOR validation annotation
    training/                                           # raw per-video JSONs (shard structure)
    validation/
```

#### Train

The training file is built directly from VidSTG's raw per-video JSONs and uses the 80 label categories available in VidSTG.

**Step 1 — Convert raw VidSTG to COCO format** :
```bash
# Edit split and path at the top of the file
python tools/data_prep/vidstg/vidstg2coco.py
# writes datasets/VidSTG/annotations/train_instances_.json
```

**Step 2 — Sub-sample up to 200 frames per video**:
```bash
python tools/data_prep/vidstg/save_vidstg_200_frames.py \
    --ann datasets/VidSTG/annotations/train_instances_.json --num_frames 200
# writes train_instances_200_frames_.json 
```

#### Val

We reproduce the annotations from [DVOC-DS](https://github.com/google-research/scenic/tree/main/scenic/projects/densecap_video) and follow their methodology.

**Step 1 — Build tfrecords** (requires tensorflow):
```bash
mkdir -p datasets/VidSTG/tfrecords

# VidOR base tfrecord (built once, used by build_vidstg_tfrecord)
python tools/data_prep/vidstg/build_vidor_tfrecord.py \
    --ann_path=datasets/VidSTG/annotations/training.json \
    --video_dir=datasets/VidSTG/video/ \
    --output_path=datasets/VidSTG/tfrecords/vidor.training.tfrecord@256

# VidSTG val tfrecord (max200f variant)
python tools/data_prep/vidstg/build_vidstg_tfrecord.py \
    --vidstg_json=datasets/VidSTG/annotations/val_annotations.json \
    --vidor_json_path=datasets/VidSTG/annotations/training/ \
    --vidor_tfrecord_path=datasets/VidSTG/tfrecords/vidor.training.tfrecord@256 \
    --video_max_len=200 \
    --output_path=datasets/VidSTG/tfrecords/vidstg_max200f_val.tfrecord@32
```

**Step 2 — Convert tfrecord to formatted JSON:**
```bash
python tools/data_prep/create_lvvis_json_from_tfrecord.py \
    --input_tfrecord=datasets/VidSTG/tfrecords/vidstg_max200f_val.tfrecord@32 \
    --output_json=datasets/VidSTG/annotations/vidstg_max200f_val_instances_.json
```
</details>

---

### VLN

[Video Localized Narratives](https://google.github.io/video-localized-narratives/) (VLN) builds on top of [UVO](https://sites.google.com/view/unidentified-video-object) sparsely annotated videos (~3 annotated frames per video). For inference we run CaptionFormer inference on an extended window of the dense video for proper tracking and filter back to the sparesly annotated frames for evaluation.

VLN has only 1 category so we use the same annotations as [DVOC-DS](https://github.com/google-research/scenic/tree/main/scenic/projects/densecap_video) for training and evaluation.

**Download (raw inputs):**
- Sparsely annotated videos (mp4) from [UVO](https://sites.google.com/view/unidentified-video-object)
- Pre-processed VLN annotations : [Google Drive folder](https://drive.google.com/drive/folders/11E86QHJW7tJw8AvlZAUEdsWs_A3DpJ7l?usp=sharing).

**Expected layout:**
```
datasets/VLN/
  uvo_videos_sparse/                                            # mp4 files
  annotations/
    vng_uvo_sparse_train_instances_.json
    vng_uvo_sparse_val_instances_.json
```

**Step 1 — Extract jpg frames** (sparse-annotated frames + offset frames `{-25,-20,-15,-10,-5,0}` needed by the model at inference):
```bash
# val
python tools/data_prep/vln/extract_jpg.py \
    --uvo_ann   datasets/VLN/annotations/vng_uvo_sparse_val_instances_.json \
    --video_dir datasets/VLN/uvo_videos_sparse \
    --frames_dir datasets/VLN/uvo_videos_sparse_frames

# train (same with train annotation)
```

**Step 2 — Build the extended val annotation file** (for val inference):
```bash
python tools/data_prep/vln/build_continuous_ann_file.py \
    --gt_json datasets/VLN/annotations/vng_uvo_sparse_val_instances_.json
# writes vng_uvo_sparse_val_extended_instances_.json next to it
```

**Step 3 — Build the COCO format annotation file for evaluation:**
```bash
python tools/data_prep/lvvis2coco_annotations.py \
    --gt_json datasets/VLN/annotations/vng_uvo_sparse_val_instances_.json
# writes vng_uvo_sparse_val_instances_coco_format.json next to the input
```

**Final expected layout:**
```
datasets/VLN/
  uvo_videos_sparse_frames/<ytid>/<frame_idx>.png       # extracted frames
  annotations/
    vng_uvo_sparse_train_instances_.json 
    vng_uvo_sparse_val_instances_.json                  # sparse 
    vng_uvo_sparse_val_extended_instances_.json         # model input extended for inference
    vng_uvo_sparse_val_instances_coco_format.json       # coco format eval annotations
```

<details>
<summary><b>Full annotation reproduction</b> — rebuild the JSONs from raw UVO + VNG (optional)</summary>

We follow [DVOC-DS](https://github.com/google-research/scenic/tree/main/scenic/projects/densecap_video) instructions to generate the annotations. 

Additional downloads :
- UVO sparse video annotations (`UVO_sparse_{train,val}_video.json`): from [UVO](https://sites.google.com/view/unidentified-video-object).
- VNG narratives + extra_masks: from [Video Localized Narratives](https://google.github.io/video-localized-narratives/).

**Raw input layout:**
```
datasets/VLN/
  uvo_videos_sparse/                                            # mp4 files
  UVO_sparse_train_video.json                                   # UVO sparse tracking ann
  UVO_sparse_val_video.json
  vng/UVO_VNG/
    meta_expressions/sparse_{train,val}/meta_expressions.json   # VNG narratives
    extra_masks/sparse_{train,val}/extra_masks.json             # VNG extra masks
```

FIrst extract the jpg frames as per **Step 1** of the main path above with `--uvo_ann datasets/VLN/UVO_sparse_val_video.json` (same for the train annotation file).

**Step 1 — Build tfrecords** (requires tensorflow):
```bash
mkdir -p datasets/VLN/tfrecords
python tools/data_prep/vln/build_vln_tfrecords.py \
    --ann_path=datasets/VLN/vng/UVO_VNG/meta_expressions/sparse_val/meta_expressions.json \
    --uvo_extra_ann_path=datasets/VLN/vng/UVO_VNG/extra_masks/sparse_val/extra_masks.json \
    --uvo_ann_path=datasets/VLN/UVO_sparse_val_video.json \
    --image_dir=datasets/VLN/uvo_videos_sparse_frames/ \
    --output_path=datasets/VLN/tfrecords/vng_uvo_sparse_val.tfrecord@32
# Same for train.
```

**Step 2 — Convert tfrecord to formatted JSON annotations:**
```bash
python tools/data_prep/create_lvvis_json_from_tfrecord.py \
    --input_tfrecord=datasets/VLN/tfrecords/vng_uvo_sparse_val.tfrecord@32 \
    --output_json=datasets/VLN/annotations/vng_uvo_sparse_val_instances_.json \
    --vln
# Same for train.
```

</details>

---

### BenSMOT

[BenSMOT](https://github.com/HengLan/SMOTer) is a person-tracking + captioning benchmark, with a single category (`person`).

**Download:**
- BenSMOT videos frames from [SMOTer](https://github.com/HengLan/SMOTer).
- Pre-processed BenSMOT annotations: from our [Google Drive folder](https://drive.google.com/drive/folders/11E86QHJW7tJw8AvlZAUEdsWs_A3DpJ7l?usp=sharing).

**Expected layout:**
```
datasets/bensmot/
  {train,test}/<category>/<seq_name>/imgs/<frame>.jpg   # extracted frames
  annotations/
    train_instances_.json
    test_instances_.json
```

**Build the COCO format annotation file for evaluation:**
```bash
python tools/data_prep/lvvis2coco_annotations.py \
    --gt_json datasets/bensmot/annotations/test_instances_.json
# writes test_instances_coco_format.json next to the input
```

**Final expected layout:**
```
datasets/bensmot/
  {train,test}/<category>/<seq_name>/imgs/<frame>.jpg
  annotations/
    train_instances_.json                  # model input — train
    test_instances_.json                   # model input — eval
    test_instances_coco_format.json        # coco format eval annotations
```

<details>
<summary><b>Full annotation reproduction</b> — rebuild the per-track JSONs from BenSMOT's annotations (optional)</summary>

Additional downloads:
- BenSMOT combined annotation files (`video_summary.json`, `instance_caption.json`, `relation.json`): from [SMOT](https://github.com/HengLan/SMOT).
- Per-sequence MOT GT (`gt/gt.txt`) ships alongside the BenSMOT frames.

**Raw input layout:**
```
datasets/bensmot/
  {train,test}/<category>/<seq_name>/imgs/<frame>.jpg   # frames
  {train,test}/<category>/<seq_name>/gt/gt.txt          # per-seq MOT GT
  video_summary.json
  instance_caption.json
  relation.json
```

**Build the per-track annotation json:**
```bash
python tools/data_prep/bensmot/bensmot_coco2lvvis.py
```

</details>

---

## Pre-extracted features

CaptionFormer is based on OVFormer, which uses **CLIP ViT-B/32 image and text features** for Open Vocabulary classification. Those features should be pre-computed for the dataset. The captioning head is based on BLIP-2, which relies on **EVA-ViT-G**. The features can also be pre-extracted for saving training compute time.

- CLIP features are **required** and referenced via `MODEL.MASK_FORMER.CLIP_IMAGE_PATH` (image features pickle) and `MODEL.MASK_FORMER.CLIP_TEXT_PATH` (text classifier `.npy`).
- BLIP-2 features are an **optional training speedup**, toggled via `MODEL.CAPTIONING_HEAD.SAVED_FEATURES`.

### CLIP image features

Output pickles live under `datasets/metadata/`. The **CLIP text classifier** are provided directly in the repo (e.g. `datasets/metadata/vidstg_simple_clip_text.npy`). You can regenerate them for VidSTG with `tools/data_prep/vidstg/save_vidstg_clip_text_embeddings.py`. For the other datasets, refer to [OVFormer](https://github.com/fanghaook/OVFormer).

To compute the CLIP image features, we provide scripts for each dataset:

```bash
# LVIS train + val
python tools/data_prep/lviscap_lvviscap/save_lvis_clip_features.py

# LV-VIS train + val + test
python tools/data_prep/lviscap_lvviscap/save_lvvis_clip_features.py

# VidSTG train + val
python tools/data_prep/vidstg/save_clip_features.py
# Since the training set is large, you can split across N jobs with `--chunk_id <ID> --num_chunks <NUM> --skip_val`,
# then merge with `--merge_chunks --num_chunks <NUM>`. Multi-GPU example:
# tools/data_prep/vidstg/save_clip_features_multigpu_example.py.

# VLN and BenSMOT have only 1 single category, so the extraction is not needed.
```

### BLIP-2 features (optional training speedup)

<details>
<summary><b>BLIP-2 feature pre-extraction</b></summary>


BLIP-2 relies on the EVA-ViT-G vision encoder, whose features can be pre-computed once per dataset to skip the forward pass at every training step.

Extraction tools:
```bash
# Video datasets (LV-VIS, VidSTG, VLN, BenSMOT)
python tools/data_prep/_feature_extraction/extract_blip2_features.py \
    --dataset {lvvis,vidstg,vln,bensmot} --split {train,val,test}

# LVIScap / LVIScap+COCO (image)
python tools/data_prep/_feature_extraction/lviscap_extract_blip2_features.py
```

Using pre-extracted features is toggled with `MODEL.CAPTIONING_HEAD.SAVED_FEATURES: True/False`.
 <!-- The matching `MODEL.CAPTIONING_HEAD.FEATURE_MAPPING` (one of `"lvis"`, `"vg"`, `"lvvis"`, `"vidstg"`, `"bensmot"`, `"vln_uvo_sparse"`) is set automatically by `train_net*.py` from `cfg.DATASETS.{TRAIN,TEST}[0]`, so you don't need to set it manually. -->

</details>