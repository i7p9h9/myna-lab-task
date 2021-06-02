#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, blank=18, **kwargs):
        super(LossFunction, self).__init__()

        self.blank = blank
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ctc = nn.CTCLoss(blank=self.blank)
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised CTC loss')

    def forward(self, x, target, target_lengths):
        
        input_lengths = torch.full(size=(x.shape[0],), fill_value=x.shape[1], dtype=torch.long)
        y = torch.tensordot(x, self.W, dims=1)
        loss = self.ctc(y.transpose(1, 0).log_softmax(2), 
                        target, input_lengths, target_lengths)
        return loss
