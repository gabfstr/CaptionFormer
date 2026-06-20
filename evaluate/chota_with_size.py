#!/usr/bin/env python3
"""
Modified CHOTA evaluator that tracks metrics per object size.
Extends the base CHOTA class to provide detailed size-based statistics.
"""
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/google-research/scenic/blob/main/scenic/projects/densevoc/chota.py

import numpy as np
from collections import defaultdict
from .chota import CHOTA
from . import mask as maskUtils


# Size thresholds (in px²)
SIZE_THRESHOLDS = {
    'small': (0, 30000),      # area < 30k
    'medium': (30000, 200000), # 30k <= area < 200k
    'large': (200000, float('inf'))  # area >= 200k
}


def compute_bbox_area(bbox):
    """
    Compute area of a bounding box.
    Handles both single bbox and list of bboxes (video format).
    Returns list of areas.
    """
    if not bbox:
        return []
    
    # Check if this is a list of bboxes (video format)
    if isinstance(bbox[0], (list, tuple)):
        areas = []
        for frame_bbox in bbox:
            if frame_bbox and len(frame_bbox) >= 4:
                try:
                    width = float(frame_bbox[2]) if frame_bbox[2] is not None else 0
                    height = float(frame_bbox[3]) if frame_bbox[3] is not None else 0
                    area = width * height
                    if area > 0:
                        areas.append(area)
                except (ValueError, TypeError):
                    continue
        return areas
    else:
        # Single bbox format
        if len(bbox) >= 4:
            try:
                width = float(bbox[2]) if bbox[2] is not None else 0
                height = float(bbox[3]) if bbox[3] is not None else 0
                area = width * height
                if area > 0:
                    return [area]
            except (ValueError, TypeError):
                pass
        return []


def get_annotation_size_category(ann):
    """
    Determine size category (small/medium/large) for an annotation.
    Returns the size category based on average bbox/mask area.
    """
    # Try bbox first
    bbox = ann.get('bboxes', ann.get('bbox', []))
    areas = compute_bbox_area(bbox)
    
    # If no bbox, try segmentation mask
    if not areas and 'segmentation' in ann:
        seg = ann['segmentation']
        try:
            # Handle RLE format segmentation
            if isinstance(seg, dict) and 'counts' in seg:
                area = maskUtils.area(seg)
                if area > 0:
                    areas = [area]
            # Handle polygon format (list of lists)
            elif isinstance(seg, list):
                # For polygon, compute bbox from coordinates
                # This is an approximation - proper handling would decode the polygon
                pass
        except Exception as e:
            pass
    
    if not areas:
        return None
    
    avg_area = np.mean(areas)
    
    for size_name, (min_area, max_area) in SIZE_THRESHOLDS.items():
        if min_area <= avg_area < max_area:
            return size_name
    
    return None


class CHOTAWithSize(CHOTA):
    """
    Extended CHOTA evaluator that tracks metrics per object size category.
    
    This class extends the base CHOTA evaluator to provide additional
    size-based breakdowns: small, medium, and large objects.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_sizes = True
        
    def compute_metrics(self, gt_data, pred_data, score_thresh=0.5, ann_format='coco', image=False):
        """
        Compute CHOTA metrics with size-based breakdowns.
        
        Returns a dictionary with:
        - Standard CHOTA metrics (overall)
        - size_breakdown: metrics per size category
        - size_stats: statistics about size distribution
        """
        # First, compute standard metrics
        results = super().compute_metrics(gt_data, pred_data, score_thresh, ann_format, image)
        
        # Add size-based analysis
        size_results = self._compute_size_breakdown(gt_data, pred_data, score_thresh, ann_format, image)
        results['size_breakdown'] = size_results
        
        return results
    
    def _compute_size_breakdown(self, gt_data, pred_data, score_thresh, ann_format, image):
        """
        Compute metrics separately for each size category.
        """
        size_results = {}
        
        # Get annotations by size
        gt_by_size = self._group_annotations_by_size(gt_data)
        pred_by_size = self._group_annotations_by_size(pred_data)
        
        # Compute statistics
        size_stats = {}
        for size_name in ['small', 'medium', 'large']:
            gt_count = len(gt_by_size.get(size_name, []))
            pred_count = len(pred_by_size.get(size_name, []))
            size_stats[size_name] = {
                'gt_count': gt_count,
                'pred_count': pred_count
            }
        
        size_results['stats'] = size_stats
        size_results['metrics'] = {}
        
        # Compute CHOTA metrics for each size
        for size_name in ['small', 'medium', 'large']:
            if size_name not in gt_by_size or len(gt_by_size[size_name]) == 0:
                # No GT annotations for this size
                size_results['metrics'][size_name] = {
                    'CHOTA': 0.0, 'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0
                }
                continue
            
            # Filter data to only this size
            gt_filtered = self._filter_data_by_size(gt_data, gt_by_size[size_name])
            pred_filtered = self._filter_data_by_size(pred_data, pred_by_size.get(size_name, []))
            
            # Handle empty predictions
            if not pred_filtered or len(self._get_annotations_list(pred_filtered)) == 0:
                size_results['metrics'][size_name] = {
                    'CHOTA': 0.0, 'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0
                }
                continue
            
            # Compute metrics for this size
            try:
                metrics = super().compute_metrics(
                    gt_filtered, pred_filtered, score_thresh, ann_format, image
                )
                size_results['metrics'][size_name] = metrics
            except Exception as e:
                print(f"Warning: Error computing metrics for {size_name} objects: {e}")
                size_results['metrics'][size_name] = {
                    'CHOTA': 0.0, 'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0
                }
        
        return size_results
    
    def _get_annotations_list(self, data):
        """Get list of annotations from data (handles both dict and list formats)."""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('annotations', [])
        return []
    
    def _group_annotations_by_size(self, data):
        """
        Group annotations by size category.
        Returns dict mapping size_name -> list of annotation indices.
        """
        annotations = self._get_annotations_list(data)
        by_size = defaultdict(list)
        
        for idx, ann in enumerate(annotations):
            size_cat = get_annotation_size_category(ann)
            if size_cat:
                by_size[size_cat].append(idx)
        
        return by_size
    
    def _filter_data_by_size(self, data, indices):
        """
        Filter data to only include annotations at given indices.
        """
        annotations = self._get_annotations_list(data)
        
        # Get annotations at specified indices
        filtered_annotations = [annotations[i] for i in indices if i < len(annotations)]
        
        # Return in same format as input
        if isinstance(data, list):
            return filtered_annotations
        else:
            # Dict format
            import copy
            filtered_data = copy.deepcopy(data)
            filtered_data['annotations'] = filtered_annotations
            return filtered_data
