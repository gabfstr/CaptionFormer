<div align="center">

# CaptionFormer: Unified Segmentation, Tracking, and Captioning for Spatio-Temporal Objects (CVPR 2026)

<a href="http://www.gabriel.fiastre.fr/"><strong>Gabriel Fiastre</strong></a>
·
<a href="https://antoyang.github.io/"><strong>Antoine Yang</strong></a>
·
<a href="https://cordeliaschmid.github.io/"><strong>Cordelia Schmid</strong></a>

[![Static Badge](https://img.shields.io/badge/arXiv-CaptionFormer-A42C25?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2510.14904)
[![Static Badge](https://img.shields.io/badge/Project-Page-388E6A?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPgo8c3ZnIHdpZHRoPSI4MDBweCIgaGVpZ2h0PSI4MDBweCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBjb2xvcj0iIzM4OEU2QSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTMgNkMzIDQuMzQzMTUgNC4zNDMxNSAzIDYgM0gxNEMxNS42NTY5IDMgMTcgNC4zNDMxNSAxNyA2VjE0QzE3IDE1LjY1NjkgMTUuNjU2OSAxNyAxNCAxN0g2QzQuMzQzMTUgMTcgMyAxNS42NTY5IDMgMTRWNloiIHN0cm9rZT0iIzM4OEU2QSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPHBhdGggZD0iTTIxIDdWMThDMjEgMTkuNjU2OSAxOS42NTY5IDIxIDE4IDIxSDciIHN0cm9rZT0iIzM4OEU2QSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPHBhdGggZD0iTTkgMTJWOEwxMi4xNDI5IDEwTDkgMTJaIiBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMzODhFNkEiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8%2BCjwvc3ZnPg%3D%3D&logoColor=red)](https://www.gabriel.fiastre.fr/captionformer/)
[![Static Badge](https://img.shields.io/badge/Scholar-CaptionFormer-3E7DFF?logo=google%20scholar&logoColor=%235D8EF6&link=https%3A%2F%2Fwww.gabriel.fiastre.fr%2Fcaptionformer%2F)](https://scholar.google.com/scholar?as_sdt=0%2C5&q=MaskCaptioner%3A+Learning+to+Jointly+Segment+and+Caption+Object+Trajectories+in+Videos&btnG=)

<img width="1200" height="384" alt="Enregistrement de l’écran 2026-06-04 à 06 06 03-2" src="https://github.com/user-attachments/assets/916f6aa0-83a6-4f4f-af29-33d4478476ec" />

</div>

## 📖 Table of Contents

[[Datasets](#-datasets--lviscap--lv-viscap)] 
[[Code](#code-coming-very-soon)] 
[[Cite](#citation)]

## 📚 Datasets — LVIScap & LV-VIScap

We release our **LVIScap** and **LV-VIScap** synthetic datasets used to train the CaptionFormer DVOC model. For more details on the synthetic caption generation, please check out the [paper](https://arxiv.org/abs/2510.14904).

⚡ **Download both annotation files** from the [Google Drive folder](https://drive.google.com/drive/folders/1qFNx7AWeXmjJ8aGiFqeO3SMoOCe6ga0D?usp=share_link).

Images and videos should be downloaded following [LVIS](https://www.lvisdataset.org/) (images, via COCO 2017) and [LV-VIS](https://github.com/haochenheheda/LVVIS) (videos).

| Dataset    | Modality | Train captions | Val captions |
|------------|----------|---------------:|-------------:|
| LVIScap    | image    |      1,244,271 |      244,083 |
| LV-VIScap  | video    |         15,966 |        3,700 |

### LVIScap

LVIScap extends [LVIS](https://www.lvisdataset.org/) with **synthetic, instance-level captions**.
Images, annotation IDs, masks, and category labels are inherited from LVIS.
The new fields per annotation are:
- `caption` — synthetic instance-level caption
- `bbox` — re-extracted as the tight bounding box of the mask

<details>
<summary><b>Annotation format</b> (click to expand)</summary>

Standard LVIS / COCO-style JSON. Vs upstream LVIS, `caption` is new and
`bbox` is re-extracted from the mask:

```jsonc
{
  "info":       { ... },
  "licenses":   [ ... ],
  "categories": [ { "id": ..., "name": ..., ... }, ... ], 
  "images": [
    {
      "id":          7,
      "width":       640,
      "height":      480,
      "coco_url":    "http://images.cocodataset.org/val2017/000000017905.jpg",
      "flickr_url":  "...",
      "license":     ...,
      "date_captured": ...,
      "neg_category_ids":            [ ... ],
      "not_exhaustive_category_ids": [ ... ]
    }
  ],
  "annotations": [
    {
      "id":           42,
      "image_id":     7,
      "category_id":  321,
      "segmentation": [ ... ],                   // polygon or RLE
      "area":         12345.0,
      "bbox":         [x, y, w, h],              // re-extracted from the mask
      "caption":      "A red leather wallet partly visible on the table."   // generated
    }
  ]
}
```

</details>

### LV-VIScap

LV-VIScap extends [LV-VIS](https://github.com/haochenheheda/LVVIS) with **synthetic, track-level captions**.
Videos, annotation IDs, per-frame masks, and category labels are inherited from LV-VIS.
The new fields per annotation are:
- `caption` — synthetic object-level caption describing the object trajectory throughout the video
- `bbox` — per-frame list of `[x, y, w, h]` tight bounding boxes extracted from each frame's mask

> LV-VIS does not provide test annotations, so we do **not** release a captioned test split. Evaluation is performed on the val set.

<details>
<summary><b>Annotation format</b> (click to expand)</summary>

YT-VIS / LV-VIS-style video JSON. Vs upstream LV-VIS, `caption` and `bbox`
are new on each track annotation:

```jsonc
{
  "licenses":   [ ... ],
  "categories": [ { "id": ..., "name": ..., ... }, ... ],
  "videos": [
    {
      "id":         7,
      "length":     36,
      "width":      1280,
      "height":     720,
      "file_names": [ "00007/00000.jpg", "00007/00001.jpg", ... ]
    }
  ],
  "annotations": [
    {
      "id":            314,
      "video_id":      7,
      "category_id":   42,
      "iscrowd":       0,
      "length":        36,
      "width":         1280, 
      "height":        720,
      "segmentations": [ rle_or_null, rle_or_null, ... ],
      "areas":         [ float_or_null, float_or_null, ... ], 
      "bbox":          [ [x, y, w, h], [x, y, w, h], ... ],   //  extracted from masks
      "caption":       "A child in a red jacket runs along the river bank."  // generated
    }
  ]
}
```

</details>

### Attribution & licensing

- LVIS: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- LV-VIS: see [upstream terms](https://github.com/haochenheheda/LVVIS)
- LVIScap & LV-VIScap synthetic captions: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), generated with **Gemini 2.0 Flash**

When using LVIScap or LV-VIScap, please cite the source dataset *and*
CaptionFormer ([bibtex](#citation)).

## Code coming (very) soon

Full code release — training, eval, demo, and
tools — is on its way. Star the repo and watch for updates. 

### Status
- [x] Dataset release (LVIScap & LV-VIScap)
- [ ] Environment setup & data preparation instructions
- [ ] Training and evaluation code
- [ ] Pretrained checkpoints
- [ ] Demo

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{fiastre2026captionformer,
  title     = {CaptionFormer: Learning to Jointly Segment and Caption Object Trajectories in Videos},
  author    = {Gabriel Fiastre and Antoine Yang and Cordelia Schmid},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
