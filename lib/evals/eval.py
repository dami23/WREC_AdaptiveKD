from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    
    return float(inter)/union

def eval_split(loader, model, split, opt, ema=None):
    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    if ema is not None:
        ema.apply_shadow()
    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0

    while True:
        data = loader.getTestBatch(split, opt)
        
        sent_ids = data['sent_ids']
        Feats = data['Feats']
        
        image_id = data['image_id']
        ann_ids = data['ann_ids']

        labels = data['labels']

        sub_wordembs = data['sub_wordembs']
        sub_classembs = data['sub_classembs']
        obj_wordembs = data['obj_wordembs']
        rel_wordembs = data['rel_wordembs']

        ann_pool5 = data['ann_pool5']
        ann_fc7 = data['ann_fc7']
        ann_fleats = data['ann_fleats']

        expand_ann_ids = data['expand_ann_ids']

        for i, sent_id in enumerate(sent_ids):
            sub_wordemb = sub_wordembs[i:i + 1]
            sub_classemb = sub_classembs[i:i + 1]
            obj_wordemb = obj_wordembs[i:i + 1]
            rel_wordemb = rel_wordembs[i:i + 1]

            label = labels[i:i + 1]
            max_len = (label != 0).sum().item()
            label = label[:, :max_len]  # (1, max_len)

            tic = time.time()
            scores, tar_attn, inter_attn, loss, sub_loss, obj_loss, rel_loss = model(Feats['pool5'], sub_wordemb, sub_classemb, obj_wordemb, rel_wordemb,
                                                               ann_pool5, ann_fc7, ann_fleats)

            scores = scores.squeeze(0)

            loss = loss.item()

            pred_ix = torch.argmax(scores)

            pred_ann_id = expand_ann_ids[pred_ix]

            gd_ix = data['gd_ixs'][i]
            loss_sum += loss
            loss_evals += 1

            pred_box = loader.Anns[pred_ann_id]['box']
            gd_box = data['gd_boxes'][i]

            IoU = computeIoU(pred_box, gd_box)
            if opt['use_IoU'] > 0:
                if IoU >= 0.5:
                    acc += 1
            else:
                if pred_ix == gd_ix:
                    acc += 1

            entry = {}
            entry['image_id'] = image_id
            entry['sent_id'] = sent_id
            entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0]  # gd-truth sent
            entry['gd_ann_id'] = data['ann_ids'][gd_ix]
            entry['pred_ann_id'] = pred_ann_id
            entry['pred_score'] = scores.tolist()[pred_ix]
            entry['IoU'] = IoU
            entry['ann_ids'] = ann_ids

            predictions.append(entry)
            toc = time.time()
            model_time += (toc - tic)

            if num_sents > 0  and loss_evals >= num_sents:
                finish_flag = True
                break
        
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if verbose:
            print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
                  (split, ix0, ix1, acc*100.0/loss_evals, loss, model_time/len(sent_ids)))

        model_time = 0

        if finish_flag or data['bounds']['wrapped']:
            break

    return loss_sum / loss_evals, acc / loss_evals, predictions
