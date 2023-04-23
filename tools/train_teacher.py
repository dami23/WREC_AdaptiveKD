from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import time
import random
import sys

# model
import _init_paths

from loaders.dataloader_refined import DataLoader
from layers.model_teacher import TeacherModel

import evals.utils as model_utils
import evals.eval as eval_utils
from opt import parse_opt
from Config import *

import torch
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(args):
    opt = vars(args)
    # initialize
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['exp_id'])
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    opt['learning_rate'] = learning_rate
    opt['eval_every'] = eval_every
    opt['learning_rate_decay_start'] = learning_rate_decay_start
    opt['learning_rate_decay_every'] = learning_rate_decay_every
    opt['word_emb_size'] = word_emb_size
    opt['class_size'] = class_size
    opt['noun_candidate_size'] = noun_candidate_size
    opt['prep_candidate_size'] = prep_candidate_size
    opt['max_iters'] = max_iters

    # set random seed
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])

    # set up loader
    data_dir = '/home/imi1214/MJP/projects/'
    data_json = osp.join(data_dir, 'cache/prepro', opt['dataset_splitBy'], 'data.json')
    data_h5 = osp.join(data_dir, 'cache/prepro', opt['dataset_splitBy'], 'data.h5')
    sub_obj_wds = osp.join(data_dir, 'cache/sub_obj_wds', opt['dataset_splitBy'], 'sub_obj_wds.json')
    similarity = osp.join(data_dir, 'cache/similarity', opt['dataset_splitBy'], 'similarity.json')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, similarity=similarity, opt=opt)

    # set up model
    opt['vocab_size'] = loader.vocab_size                              # 1999
    opt['fc7_dim'] = 2048                                             # 2048
    opt['pool5_dim'] = 1024
    opt['num_atts'] = loader.num_atts                                  # 50
    opt['interaction'] = 0                                             # 0: orginal DTWREG, 1: RAT+MMI, 2:  RAT+MMI+Adaptive ReGround
    opt['pair_feat_size'] = 1024*5
    
    model = TeacherModel(opt)

    infos = {}
    if opt['start_from'] is not None:
        pass
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)

    if opt['gpuid'] >= 0:
        model.cuda()

    lr = opt['learning_rate']

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'],
                                 weight_decay=opt['weight_decay'])

    data_time, model_time = 0, 0
    start_time = time.time()

    result_file = "./result_{}_{}.txt".format(opt['dataset_splitBy'], opt['exp_id'])
    f = open(result_file, "w")
    f.close()

    while True:
        torch.cuda.empty_cache()

        model.train()
        optimizer.zero_grad()

        T = {}

        tic = time.time()
        data = loader.getBatch('train', opt)

        sub_wordembs = data['sub_wordembs']                                          # (12, 300)
        obj_wordembs = data['obj_wordembs']                                          # (12, 300)
        rel_wordembs = data['rel_wordembs']                                          # (12, 300)

        sub_classembs = data['sub_classembs']                                        # (12, 300)

        ann_pool5 = data['ann_pool5']                                                # (12, 1024, 7, 7)
        ann_fc7 = data['ann_fc7']                                                    # (12, 1024, 7, 7)
        ann_fleats = data['ann_fleats']                                              # (12, 5)                                    

        Feats = data['Feats']

        T['data'] = time.time() - tic
        tic = time.time()
        
        scores, loss, sub_loss, obj_loss, rel_loss = model(Feats['pool5'], sub_wordembs, sub_classembs, obj_wordembs, rel_wordembs,
                                                           ann_pool5, ann_fc7, ann_fleats)

        try:
          loss.backward()

        except RuntimeError:
            continue

        model_utils.clip_gradient(optimizer, opt['grad_clip'])
        optimizer.step()

        T['model'] = time.time() - tic
        wrapped = data['bounds']['wrapped']

        data_time += T['data']
        model_time += T['model']

        total_time = (time.time() - start_time)/3600
        total_time = round(total_time, 2)

        if iter % opt['losses_log_every'] == 0:
            loss_history[iter] = loss.item()
            print('i[%s], e[%s], sub_loss=%.3f, obj_loss=%.3f, rel_loss=%.3f, lr=%.2E, time=%.3f h' % (iter, epoch, sub_loss.item(), obj_loss.item(), rel_loss.item(), lr, total_time))

            data_time, model_time = 0, 0

        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor = 0.1 ** frac
            lr = opt['learning_rate'] * decay_factor
            model_utils.set_lr(optimizer, lr)

        if (iter % opt['eval_every'] == 0) and (iter > 0) or iter == opt['max_iters']:
            val_loss, acc, predictions = eval_utils.eval_split(loader, model, 'val', opt)
            val_loss_history[iter] = val_loss
            val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
            val_accuracies += [(iter, acc)]
            print('validation loss: %.2f' % val_loss)
            print('validation acc : %.2f%%\n' % (acc * 100.0))

            current_score = acc

            f = open(result_file, "a")
            f.write(str(current_score) + "\n")
            f.close()

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_predictions = predictions
                checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
                checkpoint = {}
                checkpoint['model'] = model
                checkpoint['opt'] = opt
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)

            # write json report
            infos['iter'] = iter
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['loss_history'] = loss_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['best_val_score'] = best_val_score
            infos['best_predictions'] = predictions if best_predictions is None else best_predictions

            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            
            with open(osp.join(checkpoint_dir, opt['id'] + '.json'), 'w', encoding="utf8") as io:
                json.dump(infos, io)

        iter += 1
        if wrapped:
            epoch += 1
        if iter >= opt['max_iters'] and opt['max_iters'] > 0:
            print(str(best_val_score))
            break

if __name__ == '__main__':
    args = parse_opt()
    main(args)
