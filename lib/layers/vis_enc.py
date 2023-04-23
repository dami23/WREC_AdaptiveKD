from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        assert isinstance(bottom, Variable), 'bottom must be variable'

        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        
        return bottom_normalized_scaled

class PairEncoder(nn.Module):
    def __init__(self, opt):
        super(PairEncoder, self).__init__()
        self.word_vec_size = opt['word_vec_size']

        # location information
        self.jemb_dim = opt['jemb_dim']    # 512
        init_norm = opt.get('visual_init_norm', 20)
        self.lfeats_normalizer = Normalize_Scale(5, init_norm)
        self.fc = nn.Linear(5, opt['jemb_dim'])

        # visual information
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']       #(1024, 2048)
        self.pool5_normalizer = Normalize_Scale(opt['pool5_dim'], opt['visual_init_norm'])
        self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.att_normalizer = Normalize_Scale(opt['jemb_dim'], opt['visual_init_norm'])
        self.phrase_normalizer = Normalize_Scale(opt['word_vec_size'], opt['visual_init_norm'])
        self.att_fuse = nn.Sequential(nn.Linear(opt['pool5_dim'] + opt['fc7_dim'], opt['jemb_dim']),
                                      nn.BatchNorm1d(opt['jemb_dim']))

    def forward(self, pool5, ann_pool5, ann_fc7, ann_fleats):
        ## pool5.shape [12, 12, 1024, 7, 7], fc7.shape[12, 12, 2048, 7, 7] , ann_pool5.shape[12, 1024, 7, 7], 
        ## ann_fc7.shape [12, 2048, 7, 7] , ann_fleats.shape [12, 5]
        
        sent_num, ann_num, grids = pool5.size(0), pool5.size(1), pool5.size(3) * pool5.size(4)
        batch = sent_num * ann_num
        
        ann_pool5 = ann_pool5.contiguous().view(ann_num, self.pool5_dim, -1)
        ann_pool5 = ann_pool5.transpose(1, 2).contiguous().view(-1, self.pool5_dim)
        ann_pool5 = self.pool5_normalizer(ann_pool5)
        ann_pool5 = ann_pool5.view(ann_num, 49, -1).transpose(1, 2).contiguous().mean(2)

        expand_1_pool5 = ann_pool5.unsqueeze(1).expand(ann_num, ann_num, self.pool5_dim)
        expand_0_pool5 = ann_pool5.unsqueeze(0).expand(ann_num, ann_num, self.pool5_dim)

        expand_1_pool5 = expand_1_pool5.contiguous().view(-1, self.pool5_dim)
        expand_0_pool5 = expand_0_pool5.contiguous().view(-1, self.pool5_dim)

        expand_1_pool5 = expand_1_pool5.unsqueeze(0).expand(sent_num, ann_num*ann_num, self.pool5_dim)        # [12, 144, 1024]
        expand_0_pool5 = expand_0_pool5.unsqueeze(0).expand(sent_num, ann_num*ann_num, self.pool5_dim)

        ann_fc7 = ann_fc7.contiguous().view(ann_num, self.fc7_dim, -1)
        ann_fc7 = ann_fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)
        ann_fc7 = self.fc7_normalizer(ann_fc7)
        ann_fc7 = ann_fc7.view(ann_num, 49, -1).transpose(1, 2).contiguous().mean(2)

        expand_1_fc7 = ann_fc7.unsqueeze(1).expand(ann_num, ann_num, self.fc7_dim)
        expand_0_fc7 = ann_fc7.unsqueeze(0).expand(ann_num, ann_num, self.fc7_dim)

        expand_1_fc7 = expand_1_fc7.contiguous().view(-1, self.fc7_dim)
        expand_0_fc7 = expand_0_fc7.contiguous().view(-1, self.fc7_dim)

        expand_1_fc7 = expand_1_fc7.unsqueeze(0).expand(sent_num, ann_num*ann_num, self.fc7_dim)             # [12, 144, 2048]
        expand_0_fc7 = expand_0_fc7.unsqueeze(0).expand(sent_num, ann_num*ann_num, self.fc7_dim)

        ann_fleats = self.lfeats_normalizer(ann_fleats.contiguous().view(-1, 5))
        ann_fleats = self.fc(ann_fleats)

        expand_1_fleats = ann_fleats.unsqueeze(1).expand(ann_num, ann_num, 512)
        expand_0_fleats = ann_fleats.unsqueeze(0).expand(ann_num, ann_num, 512)

        expand_1_fleats = expand_1_fleats.contiguous().view(-1, 512)
        expand_0_fleats = expand_0_fleats.contiguous().view(-1, 512)

        expand_1_fleats = expand_1_fleats.unsqueeze(0).expand(sent_num, ann_num * ann_num, 512)              # [12, 144, 512]
        expand_0_fleats = expand_0_fleats.unsqueeze(0).expand(sent_num, ann_num * ann_num, 512)

        pair_feats = torch.cat([expand_1_fc7, expand_1_fleats, expand_0_fc7, expand_0_fleats], 2)            # [12, 144, 5120]
        
        return pair_feats, expand_1_pool5, expand_1_fc7, expand_1_fleats, expand_0_pool5, expand_0_fc7, expand_0_fleats
