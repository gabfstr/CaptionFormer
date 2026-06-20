# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/HengLan/SMOT/blob/main/eval_vu.py

import json
import numpy as np
import os

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from argparse import ArgumentParser


def eval_language(preds, gts, task):
    res = {}
    # 计算BLEU得分
    bleu_score = corpus_bleu(gts, preds)
    print(str(task), "BLEU score:", bleu_score)
    res['bleu'] = bleu_score

    # 计算METEOR得分
    assert len(gts) == len(preds)
    num, m_score = len(gts), 0.0
    for i in range(num):
        me_gt = gts[i][0].replace(',', '').replace('.', '')#.split(' ')
        me_pred = preds[i].replace(',', '').replace('.', '')#.split(' ')
        # print("me_gt", me_gt)
        # print("me_pred", me_pred)
        # m_score += meteor_score([me_gt], me_pred)
        m_score += meteor_score([[me_gt]], [me_pred])
    print(str(task), "METEOR score:", m_score / num)
    res['meteor'] = m_score / num

    preds_dict, gts_dict = {}, {}
    for i in range(len(gts)):
        preds_dict[int(i)] = [preds[i]]
        gts_dict[int(i)] = gts[i]

    # 计算ROUGE-N、ROUGE-L和ROUGE-W得分
    rouge_eval = Rouge()
    rouge_score, _ = rouge_eval.compute_score(gts_dict, preds_dict)
    print(str(task), "ROUGE score:", rouge_score)
    res['rouge'] = rouge_score

    # 计算 CIDEr 分数
    cider_eval = Cider()
    cider_score, _ = cider_eval.compute_score(gts_dict, preds_dict)
    print(str(task), "CIDEr score:", cider_score)
    res['cider'] = cider_score
    
    return res


def eval_relation(preds, gts):
    # 计算准确率
    accuracy = accuracy_score(gts, preds)
    print("Accuracy:", accuracy)

    # 计算精确率
    precision = precision_score(gts, preds, average='micro')
    print("Precision:", precision)

    # 计算召回率
    recall = recall_score(gts, preds, average='micro')
    print("Recall:", recall)

    # 计算F1值
    f1 = f1_score(gts, preds, average='micro')
    print("F1 Score:", f1)


def main(args):
    root_path = args.res_folder
    summary_path = os.path.join(root_path,'summary_results.json')
    caption_path = os.path.join(root_path,'caption_results.json')
    relation_path = os.path.join(root_path,'relation_results.json')
    if not args.caption_only:
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        with open(relation_path, 'r') as f:
            relation_data = json.load(f)
    else:
        summary_data = {}
        relation_data = {}

    with open(caption_path, 'r') as f:
        caption_data = json.load(f)
    
    
    # summary_preds = summary_data['preds']
    # summary_gts = summary_data['gts']
    # summary_gts = [[gt] for gt in summary_gts]
    
    if not args.caption_only:
        summary_preds = []
        for k, x in summary_data.items():
            for pred in x['pred']:
                summary_preds.append(pred)
        summary_gts = [x["gt"] for k,x in summary_data.items()]

        if len("summary_preds") == 0:
            print("No summary preds. Specify the correct path, or caption_only")
            return

        eval_language(summary_preds, summary_gts, task='summary')
    
    caption_preds=[]
    caption_gts =  []
    for k, x in caption_data.items():
        # print("k", k)
        # print("x", x)
        # raise ValueError
        for i, gt in enumerate(x['gt']):
            if gt is not None:
                caption_gts.append([gt])
                caption_preds.append(x['pred'][i])
        # for pred in x['pred']:
        #     caption_preds.append(pred)
        # for gt in x['gt']:
        #     caption_gts.append([gt])
        
    # caption_preds = caption_data['preds']
    # caption_gts = caption_data['gts']
    # caption_gts = [[gt] for gt in caption_gts]
    
    cap_res = eval_language(caption_preds, caption_gts, task='caption')

    if args.store:
        store_path = os.path.join(root_path,'summary.txt')
        with open(store_path, 'w') as f:
            for k, v in cap_res.items():
                f.write(f"{k}: {v}\n")
        print("Stored results in", store_path)

    if not args.caption_only:
        # relation_preds = relation_data['preds']
        # relation_gts = relation_data['gts']
        relation_preds = []
        relation_gts = []
        for k, x in relation_data.items():
            for pred in x['pred']:
                relation_preds.append(pred)
            for gt in x['gt']:
                relation_gts.append(gt)
        filtererd_relation_preds = []
        for relation_pred in relation_preds:
            filtererd_relation_pred = [1 if x > 0.4 else 0 for x in relation_pred]
            filtererd_relation_preds.append(filtererd_relation_pred)
        eval_relation(filtererd_relation_preds,relation_gts )

def check():
    gts = ['a girl dressed in white and black, dancing in the center.'.replace(',', '').replace('.', '').split(' ')]
    preds = 'a girl dressed in white and black, dancing in the center.'.replace(',', '').replace('.', '').split(' ')
    m_score = meteor_score(gts, preds)
    print("METEOR score:", m_score)


if __name__ == '__main__':
    # nltk.download('wordnet')

    parser = ArgumentParser()
    parser.add_argument('--res_folder', type=str, default='output/pred_results/')
    parser.add_argument('-caption_only', action='store_true')
    parser.add_argument('-store', action='store_true')
    args = parser.parse_args()
    main(args)
    # check()
