# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import argparse
import json
from lvis import LVIS, LVISEval, LVISResults

def evaluate_lvis(gt_file, dt_file):
    # Load ground truth and detections
    lvis_gt = LVIS(gt_file)
    lvis_dt = LVISResults(lvis_gt, dt_file)

    # Run evaluation
    lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
    lvis_eval.evaluate()
    lvis_eval.accumulate()
    lvis_eval.summarize()
    
    # Print per-category AP
    lvis_eval.summarize_per_cat()
    res_ap = lvis_eval.results["AP_per_cat"]
    res_ap_50 = lvis_eval.results["AP50_per_cat"]
    res_ap_75 = lvis_eval.results["AP75_per_cat"]
    res_aps = lvis_eval.results["APs_per_cat"]
    res_apm = lvis_eval.results["APm_per_cat"]
    res_apl = lvis_eval.results["APl_per_cat"]
    categories = lvis_gt.load_cats(lvis_gt.get_cat_ids())
    
    print("\nCategory-wise AP:")
    for cat, ap, ap_50, ap_75, aps, apm, apl in zip(categories, res_ap, res_ap_50, res_ap_75, res_aps, res_apm, res_apl):
        print(f"{cat['name']}, freq {cat['instance_count']} : {ap:.2f} (AP), {ap_50:.2f} (AP_50), {ap_75:.2f} (AP_75), {aps:.2f} (APs), {apm:.2f} (APm), {apl:.2f} (APl)")
    print("\n")
    print("AP : {}".format(lvis_eval.results["AP"]))
    print("AP50 : {}".format(lvis_eval.results["AP50"]))
    print("AP75 : {}".format(lvis_eval.results["AP75"]))
    print("APs : {}".format(lvis_eval.results["APs"]))
    print("APm : {}".format(lvis_eval.results["APm"]))
    print("APl : {}".format(lvis_eval.results["APl"]))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LVIS detections and print AP per category.")
    parser.add_argument("--gt", type=str, help="Path to ground truth LVIS JSON file.")
    parser.add_argument("--dt", type=str, help="Path to detections LVIS JSON file.")
    args = parser.parse_args()

    evaluate_lvis(args.gt, args.dt)
