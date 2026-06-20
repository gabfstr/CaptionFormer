# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import json 
import tqdm

# from pycocotools import mask as maskUtils
# from skimage import measure
# import numpy as np

# def rle_to_polygon(rle):
#     # Decode RLE to binary mask
#     mask = maskUtils.decode(rle)

#     # Find contours (0.5 threshold for better boundary accuracy)
#     contours = measure.find_contours(mask, 0.5)

#     polygons = []
#     for contour in contours:
#         # Flip (y, x) to (x, y), and flatten
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()

#         # Keep only valid polygons (at least 6 coordinates = 3 points)
#         if len(segmentation) >= 6:
#             polygons.append(segmentation)
    
#     return polygons

# run from root or edit path
catinfo_path = './tools/data_prep/lviscap_lvviscap/coco2lvis_cat_info.json'
lvis_ann_path = './datasets/lvis/lviscap_v1_train.json'
coco_ann_path = './datasets/coco/instances_train2017.json'
out_path = './datasets/lvis/lviscap_v1_train+coco.json'

with open(catinfo_path, 'r') as f:
    catinfo_map = json.load(f)
print("catinfo map : ", catinfo_map)
print("loading coco and lvis json files")
with open(lvis_ann_path, 'r') as f:
    lvis_ann = json.load(f)

with open(coco_ann_path, 'r') as f:
    coco_ann = json.load(f)

image_data = lvis_ann['images']
lvis_image_ids = [int(img['id']) for img in image_data]
max_id = max(lvis_image_ids)
print("max id in lvis ann :", max_id)

coco_images = coco_ann['images']

keep_og_id = False
if keep_og_id:
    for img in coco_images:
        if img['id'] in lvis_image_ids:
            # print("Image id {} already exists in lvis ann".format(img['id']))
            continue
        img_rec = {
            'id': img['id'],
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'width': img['width'],
            'height': img['height'],
            'license': img['license'],
            'date_captured': img['date_captured'],
            'file_name': 'train2017/' + img['file_name'],
        }
        image_data.append(img_rec)
else :
    img_id_counter=max_id
    img_id_map = {}
    for img in coco_images:
        img_id_counter += 1
        img_id_map[img['id']] = img_id_counter
        img_rec={'id':img_id_counter,
                'neg_category_ids':[],
                'not_exhaustive_category_ids':[],
                'width':img['width'],
                'height':img['height'],
                'license':img['license'],
                'date_captured':img['date_captured'],
                'file_name':'train2017/'+img['file_name'],
        }
        image_data.append(img_rec)
print("Added {} images to lvis ann".format(len(coco_images)))


annotations_data = lvis_ann['annotations']
ann_ids = [int(ann['id']) for ann in annotations_data]
max_ann_id = max(ann_ids)

# list_ann_counter = 0
print("Adding coco annotations to lvis anns")
ann_id_counter=max_ann_id
for ann in tqdm.tqdm(coco_ann['annotations']):
    ann_id_counter += 1
    
    segm = ann['segmentation']
    if ann['iscrowd'] == 1:
        continue    

    if isinstance(segm, dict):
        print("segmentation is dict")
        print("segm : ", segm)
        print("iscrowd : ", ann['iscrowd'])
        print("image_id : ", ann['image_id'])
        raise ValueError("Segmentation is RLE, please convert to list of polygons")
        # list_ann_counter += 1

    
    ann_rec = {
        'id': ann_id_counter,
        'image_id': img_id_map[ann['image_id']],
        'category_id': catinfo_map[str(ann['category_id'])],
        'iscrowd': 0,
        'area': ann['area'],
        'bbox': ann['bbox'],
        'segmentation': segm,
    }
    annotations_data.append(ann_rec)
# print("list ann counter : ", list_ann_counter)
# raise Exception("segmentation is dict, please convert to list")
    
print("Added {} annotations to lvis ann".format(len(coco_ann['annotations'])))

final_data = {
    'info': lvis_ann['info'],
    'categories': lvis_ann['categories'],
    'images': image_data,
    'annotations': annotations_data,
    'licenses': lvis_ann['licenses'],
}

with open(out_path, 'w') as f:
    json.dump(final_data, f)
print("Saved to ", out_path)
print("Total images : ", len(image_data))
print("Total annotations : ", len(annotations_data))



