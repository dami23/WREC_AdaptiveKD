from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from layers.vis_enc import PairEncoder
import pdb

class SimAttention(nn.Module):
    def __init__(self, vis_dim, words_dim, jemb_dim):
        super(SimAttention, self).__init__()
        self.embed_dim = 300
        self.words_dim = words_dim
        self.feat_fuse = nn.Sequential(nn.Linear(words_dim + vis_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, jemb_dim),
                                       nn.ReLU(),
                                       nn.Linear(jemb_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_emb, vis_feats):
        sent_num, ann_num  = vis_feats.size(0), vis_feats.size(1)
        word_emb = word_emb.unsqueeze(1).expand(sent_num, ann_num, self.words_dim)
        word_emb_n = nn.functional.normalize(word_emb, p=2, dim=2)
        
        sim_attn = self.feat_fuse(torch.cat([word_emb_n, vis_feats], 2))
        sim_attn_n = self.softmax(sim_attn.view(sent_num, ann_num))
        # sim_attn = sim_attn.squeeze(2)

        return sim_attn_n

class TeacherModel(nn.Module):
    def __init__(self, opt):
        super(TeacherModel, self).__init__()
        self.num_layers = opt['rnn_num_layers']
        self.hidden_size = opt['rnn_hidden_size']
        self.num_dirs = 2 if opt['bidirectional'] > 0 else 1
        self.jemb_dim = opt['jemb_dim']
        self.word_vec_size = opt['word_vec_size']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.sub_filter_type = opt['sub_filter_type']
        self.filter_thr = opt['sub_filter_thr']
        self.word_emb_size = opt['word_emb_size']

        # new objects
        self.visual_noun = nn.Sequential(
            nn.Linear(self.fc7_dim, 1024),
            # nn.Linear(self.pool5_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.Linear(512, opt['noun_candidate_size']),
            nn.Linear(512, opt['class_size']),
            #nn.ReLU(),
        )

        self.visual_emb = nn.Sequential(
            nn.Linear(self.fc7_dim, 1024),
            # nn.Linear(self.pool5_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, opt['word_emb_size']),
            #nn.ReLU(),
        )

        self.pair_emb = nn.Sequential(
            nn.Linear(opt['pair_feat_size'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, opt['word_emb_size']),
            #nn.ReLU(),
        )

        self.pair_encoder = PairEncoder(opt)

        self.pair_attn = SimAttention(opt['pair_feat_size'], self.word_emb_size*3, self.jemb_dim)
        self.sub_attn = SimAttention(self.fc7_dim, self.word_emb_size, self.jemb_dim)

        self.mse_loss = nn.MSELoss()
        # self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, pool5, sub_wordembs, sub_classembs, obj_wordembs, rel_wordembs, ann_pool5, ann_fc7, ann_fleats):
        sent_num = pool5.size(0)
        ann_num =  pool5.size(1)
        
        sub_fuseembs = 0.1*sub_wordembs + 0.9*sub_classembs
        
        pair_wordembs = torch.cat([sub_fuseembs, obj_wordembs, rel_wordembs], 1)              # [12, 900]
        pair_feats, expand_1_pool5, expand_1_fc7, expand_1_fleats, expand_0_pool5, expand_0_fc7, expand_0_fleats = self.pair_encoder(pool5, ann_pool5, ann_fc7, ann_fleats)
        
        pair_attn = self.pair_attn(pair_wordembs, pair_feats)                                 # [12, 144]
        sub_attn = self.sub_attn(sub_fuseembs, expand_1_fc7)                                  # [12, 144]
        obj_attn = self.sub_attn(obj_wordembs, expand_0_fc7)                                  # [12, 144]
        
        re_pair_feats = torch.matmul(pair_attn.view(sent_num, 1, ann_num*ann_num), pair_feats)  # [12, 1, 5120]
        re_pair_feats = re_pair_feats.reshape([sent_num, -1])

        re_sub_feats = torch.matmul(sub_attn.view(sent_num, 1, ann_num*ann_num), expand_1_fc7)  # [12, 1, 2048]
        re_sub_feats = re_sub_feats.reshape([sent_num, -1])

        re_obj_feats = torch.matmul(obj_attn.view(sent_num, 1, ann_num * ann_num), expand_0_fc7) # [12, 1, 2048]
        re_obj_feats = re_obj_feats.reshape([sent_num, -1])

        sub_result = self.visual_emb(re_sub_feats)      # [12, 300]
        obj_result = self.visual_emb(re_obj_feats)
        rel_result = self.pair_emb(re_pair_feats)
        
        sub_loss = self.mse_loss(sub_result, sub_classembs)
        sub_loss_sum = torch.sum(sub_loss)

        # sub_cos_loss = self.cos_sim(sub_result, sub_classembs)
        # sub_loss_sum = torch.sum(sub_cos_loss)
        
        obj_loss = self.mse_loss(obj_result, obj_wordembs)
        obj_loss_sum = torch.sum(obj_loss)

        # obj_cos_loss = self.cos_sim(obj_result, obj_wordembs)
        # obj_loss_sum = torch.sum(obj_cos_loss)

        rel_loss = self.mse_loss(rel_result, rel_wordembs)
        rel_loss_sum = torch.sum(rel_loss)

        # rel_cos_loss = self.cos_sim(rel_result, rel_wordembs)
        # rel_loss_sum = torch.sum(rel_cos_loss)

        loss_sum = 1*sub_loss_sum + 1*obj_loss_sum + 1*rel_loss_sum

        final_attn = 2*sub_attn + 1*obj_attn + 1*pair_attn                            # [12, 144]
        
        return final_attn, 2*sub_attn, 1*obj_attn + 1*pair_attn, loss_sum, sub_loss_sum, obj_loss_sum, rel_loss_sum
