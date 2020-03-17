# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

def getbonejs(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for i in range(17):
        #for j in range(i+1,17) is also OK
        for j in range(i,17):
            if not ([i,j] in boneindex or [j,i] in boneindex):
                bone.append(seq[:,j] - seq[:,i])
    bone = torch.stack(bone,1)
    bone = bone.view(bs,ss, bone.size(1),3)
    return bone


def getbonelength(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])
    bone = torch.stack(bone,1)
    bone = torch.pow(torch.pow(bone,2).sum(2),0.5)
    bone = bone.view(bs,ss, bone.size(1))
    return bone


def getbonedirect(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])
    bonedirect = torch.stack(bone,1)
    bonesum = torch.pow(torch.pow(bonedirect,2).sum(2), 0.5).unsqueeze(2)
    bonedirect = bonedirect/bonesum
    bonedirect = bonedirect.view(bs,-1,3)
    return bonedirect
