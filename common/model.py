# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch
torch.backends.cudnn.benchmark=True
from torch.autograd import Variable
from common.bone import *

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        
        # Validate input
        #for fw in filter_widths:
        #    assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
        
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        
        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self._forward_blocks(x)
        
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        
        return x    




class TemporalModel(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out, boneindex, temperature, randnumtest, filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        num_joints_out-1 -- number of input bones (e.g. 16 for Human3.6M)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        # Bottom layers
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        self.expand_convscore = nn.Conv1d(num_joints_in, channels, filter_widths[0], bias=False)
        self.expand_bnscore = nn.BatchNorm1d(channels, momentum=0.1)
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        self.causal_shift2 = [ (filter_widths[0]) // 2 if causal else 0 ]
        self.causal = causal

        # The first sub-network of bone direction prediction network
        layers_conv = []
        layers_bn = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            self.causal_shift2.append((filter_widths[i]//2) if causal else 0)
            if i==1:
                layers_conv.append(nn.Conv1d(channels*2, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            else:
                layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        # The second sub-network of bone direction prediction network
        next_dilation = filter_widths[0]
        layers_conv1 = []
        layers_bn1 = []
        for i in range(1, len(filter_widths)):
            layers_conv1.append(nn.Conv1d(channels*2, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn1.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv1.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn1.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_conv1 = nn.ModuleList(layers_conv1)
        self.layers_bn1 = nn.ModuleList(layers_bn1)

        # The bone length prediction network        
        layers_convbone = []
        layers_bnbone = []
        for i in range(len(filter_widths)*2):
            if i==0:
                layers_convbone.append(nn.Linear(num_joints_in*in_features,channels))
                layers_bnbone.append(nn.BatchNorm1d(channels, momentum=0.1))
            elif i==len(filter_widths)*2-1:
                layers_convbone.append(nn.Linear(channels,num_joints_out*3))
            else:
                layers_convbone.append(nn.Linear(channels,channels))
                layers_bnbone.append(nn.BatchNorm1d(channels, momentum=0.1))

        self.layers_convbone = nn.ModuleList(layers_convbone)
        self.layers_bnbone = nn.ModuleList(layers_bnbone)
                
        
        # The bone length attention module
        self.boneatt = nn.Linear(num_joints_out*6,num_joints_out-1)
        self.softmax = nn.Softmax(dim=1)

        # Uppper layers
        self.shrink_direct_1 = nn.Conv1d(channels, (num_joints_out-1)*3, 1)
        self.shrink_direct_2 = nn.Conv1d(channels, (num_joints_out-1)*3, 1)
        self.lengthlinear = nn.Linear(num_joints_out-1,channels)
        self.directlinear = nn.Conv1d((num_joints_out-1)*3, channels, 1)
        self.lengthlinear_1 = nn.Linear(num_joints_out-1,channels)
        self.lengthlinear_2 = nn.Linear(num_joints_out-1,channels)
        self.directlinear_1 = nn.Conv1d((num_joints_out-1)*3, channels, 1)
        self.directlinear_2 = nn.Conv1d((num_joints_out-1)*3, channels, 1)
        self.jointshiftnum = int(num_joints_out*(num_joints_out+1)/2 - (num_joints_out-1))
        self.shrink_js_1 = nn.Conv1d(channels, self.jointshiftnum*3, 1)
        self.shrink_js_2 = nn.Conv1d(channels, self.jointshiftnum*3, 1)
        

        # Other parameters
        self.boneindex = boneindex
        self.temperature = temperature
        self.num_joints_out = num_joints_out
        self.num_joints_in = num_joints_in
        self.channels = channels
        self.filter_widths = filter_widths
        self.sinrecfield = int((next_dilation-1)/2)
        self.randnumtest = randnumtest
    def _forward_blocks(self, x):
        xtemp = x.view(x.size(0),self.num_joints_in,3,x.size(2))
        #visibility score feature
        xscore = xtemp[:,:,2,:]
        x = xtemp[:,:,:2,:].contiguous().view(x.size(0),self.num_joints_in*2,x.size(2))

        #first deal with the bone length prediction network
        x_rand = x.permute(0,2,1).contiguous()
        bs = x_rand.size(0)
        ss = x_rand.size(1)
        x_rand = x_rand.view(x_rand.size(0)*x_rand.size(1),-1)
        x_rand = self.drop(self.relu(self.layers_bnbone[0](self.layers_convbone[0](x_rand))))
        #independently predict the 3D joint locations of each frame of a video
        for i in range(1,len(self.pad)):
            res_rand = x_rand
            x_rand = self.drop(self.relu(self.layers_bnbone[2*i-1](self.layers_convbone[2*i-1](x_rand))))
            x_rand = self.drop(self.relu(self.layers_bnbone[2*i](self.layers_convbone[2*i](x_rand))))
            x_rand = x_rand + res_rand
        x_rand = self.layers_convbone[-1](x_rand)
        x_rand = x_rand.view(bs,ss,-1)
        # let's compute the bone length weights
        x_rand_abs = torch.abs(x_rand)
        x_rand_con = torch.cat((x_rand, x_rand_abs),2)
        x_rand_boneatt = self.boneatt(x_rand_con.view(bs*ss,-1)).view(x_rand_con.size(0), x_rand_con.size(1), -1)
        x_rand_boneatt = x_rand_boneatt * self.temperature
        bone = getbonelength(x_rand.view(bs,ss,-1,3), self.boneindex)
        # if not causal mode, predicted the bone length of the current frame as the weighted average of all the video frames
        if not self.causal:
            x_rand_boneatt = x_rand_boneatt[:,self.sinrecfield:-self.sinrecfield]
            x_rand_boneatt = self.softmax(x_rand_boneatt)        
            # get the final predicted bone length of the current frame
            bonelength = (bone[:,self.sinrecfield:-self.sinrecfield] * x_rand_boneatt).sum(1)
            bonelength = bonelength.unsqueeze(2).repeat(1,1,x.size(2)-2*self.sinrecfield)
        # in causal mode, you can only observe the frames before the current one, ranomly sample 'randnumtest' frames to predict the bone lengths
        else:
            bonelength = []
            #predict the bone length frame by frame from the (self.sinrecfield*2+1)-th frame
            for ii in range(self.sinrecfield*2, x.size(2)):
                #get the frames you can observe
                bonecausal = bone[:,self.sinrecfield*2:ii+1,:]
                perm = torch.randperm(bonecausal.size(1))
                idx = perm[:self.randnumtest]
                bonecausal = bonecausal[:,idx,:]
                x_rand_boneattcausal = self.softmax(x_rand_boneatt[:,self.sinrecfield*2:ii+1,:][:,idx,:])
                bonelengthcausal = (bonecausal * x_rand_boneattcausal).sum(1)
                bonelength.append(bonelengthcausal.unsqueeze(2))
            bonelength = torch.cat(bonelength,2)
        # Now deal with the bone direction prediction network
        #first sub-network, the prediction can be made in a parallel way
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        xbottom = [x]        
        xscore = self.drop(self.relu(self.expand_bnscore(self.expand_convscore(xscore))))
        x = torch.cat((x, xscore*x),1)
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :self.channels, pad + shift : x.shape[2] - pad + shift]
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
            xbottom.append(x)
        #second sub-network, predict frame by frame
        xall = []
        xtest = x
        for ii in range(self.sinrecfield, xtest.size(2)+self.sinrecfield):
            x = xtest[:,:,ii-self.sinrecfield:ii-self.sinrecfield+1].repeat(1,1,int((self.sinrecfield*2+1)/self.filter_widths[0]))
            for i in range(len(self.pad) - 1):
                ind = torch.LongTensor(list(range(ii-self.sinrecfield,xbottom[i].size(2),self.pad[i+1])))[:self.pad[len(self.pad)-i-1]]
                x = torch.cat((x, xbottom[i][:,:,ind]),1)
                res = x[:, :, self.causal_shift2[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
                x = self.drop(self.relu(self.layers_bn1[2*i](self.layers_conv1[2*i](x))))
                x = res[:,:self.channels] + self.drop(self.relu(self.layers_bn1[2*i + 1](self.layers_conv1[2*i + 1](x))))                
            x = self.shrink_direct_2(x)
            xall.append(x)
        x = torch.cat(xall,2)
        bonedirect = x.view(x.size(0),self.num_joints_out-1,3,x.size(2))
        bonesum = torch.pow(torch.pow(bonedirect,2).sum(2),0.5).unsqueeze(2)
        bonedirect = bonedirect/bonesum
        bonedirect = bonedirect.view(bonedirect.size(0),(self.num_joints_out-1)*3,bonedirect.size(3))
        #compute the final 3D joint locations
        bonelength = bonelength.permute(0,2,1).contiguous().view(-1,self.num_joints_out-1)
        bonel = self.lengthlinear(bonelength).view(x.size(0),-1,self.channels)
        bonel = bonel.permute(0,2,1).contiguous()
        boned = self.directlinear(bonedirect)
        bonel = bonel.view(boned.size())
        x = boned * bonel
        x = self.shrink(x)
        return x





class TemporalModelOptimized1f(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out, boneindex, temperature, 
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        num_joints_out-1 -- number of input bones (e.g. 16 for Human3.6M)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        # Bottom layers
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        self.expand_convscore = nn.Conv1d(num_joints_in, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        self.expand_bnscore = nn.BatchNorm1d(channels, momentum=0.1)
        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ]
        self.causal = causal

        # The first sub-network of bone direction prediction network
        layers_conv = []
        layers_bn = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)
            if i==1:
                layers_conv.append(nn.Conv1d(channels*2, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            else:
                layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        # The second sub-network of bone direction prediction network
        next_dilation = filter_widths[0]
        layers_conv1 = []
        layers_bn1 = []
        for i in range(1, len(filter_widths)):
            layers_conv1.append(nn.Conv1d(channels*2, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn1.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv1.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn1.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_conv1 = nn.ModuleList(layers_conv1)
        self.layers_bn1 = nn.ModuleList(layers_bn1)
        

        # The bone length prediction network
        layers_convbone = []
        layers_bnbone = []
        for i in range(len(filter_widths)*2):
            if i==0:
                layers_convbone.append(nn.Linear(num_joints_in*in_features,channels))
                layers_bnbone.append(nn.BatchNorm1d(channels, momentum=0.1))
            elif i==len(filter_widths)*2-1:
                layers_convbone.append(nn.Linear(channels,num_joints_out*3))
            else:
                layers_convbone.append(nn.Linear(channels,channels))
                layers_bnbone.append(nn.BatchNorm1d(channels, momentum=0.1))

        self.layers_convbone = nn.ModuleList(layers_convbone)
        self.layers_bnbone = nn.ModuleList(layers_bnbone)

        # The bone length attention module
        self.boneatt = nn.Linear(num_joints_out*6,num_joints_out-1)
        self.softmax = nn.Softmax(dim=1)

        # Uppper layers
        self.shrink_direct_1 = nn.Conv1d(channels, (num_joints_out-1)*3, 1)
        self.shrink_direct_2 = nn.Conv1d(channels, (num_joints_out-1)*3, 1)
        self.lengthlinear = nn.Linear(num_joints_out-1,channels)
        self.directlinear = nn.Conv1d((num_joints_out-1)*3, channels, 1)
        self.lengthlinear_1 = nn.Linear(num_joints_out-1,channels)
        self.lengthlinear_2 = nn.Linear(num_joints_out-1,channels)
        self.directlinear_1 = nn.Conv1d((num_joints_out-1)*3, channels, 1)
        self.directlinear_2 = nn.Conv1d((num_joints_out-1)*3, channels, 1)
        self.jointshiftnum = int(num_joints_out*(num_joints_out+1)/2 - (num_joints_out-1))
        self.shrink_js_1 = nn.Conv1d(channels, self.jointshiftnum*3, 1)
        self.shrink_js_2 = nn.Conv1d(channels, self.jointshiftnum*3, 1)

        # Other parameters
        self.boneindex = boneindex
        self.temperature = temperature
        self.num_joints_out = num_joints_out
        self.num_joints_in = num_joints_in
        self.channels = channels
        self.filter_widths = filter_widths
        self.sinrecfield = int((next_dilation-1)/2)
    def forward(self, x, x_rand, x_randaug):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        xtemp = x.view(x.size(0),self.num_joints_in,3,x.size(2))
        #visibility score feature
        xscore = xtemp[:,:,2,:]
        x = xtemp[:,:,:2,:].contiguous().view(x.size(0),self.num_joints_in*2,x.size(2))

        bs = x_rand.size(0)
        ss = x_rand.size(1)

        #first deal with the bone length prediction network
        x_rand = x_rand.view(x_rand.size(0)*x_rand.size(1),-1)
        x_rand = self.drop(self.relu(self.layers_bnbone[0](self.layers_convbone[0](x_rand))))
        #independently predict the 3D joint locations of each randomly sampled frame
        for i in range(1,len(self.pad)):
            res_rand = x_rand
            x_rand = self.drop(self.relu(self.layers_bnbone[2*i-1](self.layers_convbone[2*i-1](x_rand))))
            x_rand = self.drop(self.relu(self.layers_bnbone[2*i](self.layers_convbone[2*i](x_rand))))
            x_rand = x_rand + res_rand
        x_rand = self.layers_convbone[-1](x_rand)
        x_rand = x_rand.view(bs,ss, self.num_joints_out, 3)
        # let's compute the bone length weights
        x_rand2 = x_rand.view(bs,ss,-1)
        x_rand_abs = torch.abs(x_rand2.detach())
        x_rand_con = torch.cat((x_rand2.detach(), x_rand_abs),2)
        x_rand_boneatt = self.boneatt(x_rand_con.view(bs*ss,-1)).view(x_rand_con.size(0), x_rand_con.size(1), -1)
        x_rand_boneatt = x_rand_boneatt * self.temperature
        x_rand_boneatt = self.softmax(x_rand_boneatt)  
        # manually derive bone length based on the 3D joint location predictions 
        bone = getbonelength(x_rand2.detach().view(bs,ss,-1,3), self.boneindex)
        # get the final predicted bone length of the current frame
        bonelength = (bone * x_rand_boneatt).sum(1)

        # deal with the augmented data
        x_randaug = x_randaug.view(x_randaug.size(0)*x_randaug.size(1),-1)
        x_randaug = self.drop(self.relu(self.layers_bnbone[0](self.layers_convbone[0](x_randaug))))
        for i in range(1,len(self.pad)):
            res_randaug = x_randaug
            x_randaug = self.drop(self.relu(self.layers_bnbone[2*i-1](self.layers_convbone[2*i-1](x_randaug))))
            x_randaug = self.drop(self.relu(self.layers_bnbone[2*i](self.layers_convbone[2*i](x_randaug))))
            x_randaug = x_randaug + res_randaug
        x_randaug = self.layers_convbone[-1](x_randaug)
        x_randaug = x_randaug.view(bs,ss, self.num_joints_out, 3)
        x_randaug2 = x_randaug.view(bs,ss,-1)
        x_rand_absaug = torch.abs(x_randaug2.detach())
        x_rand_conaug = torch.cat((x_randaug2.detach(), x_rand_absaug),2)
        x_rand_boneattaug = self.boneatt(x_rand_conaug.view(bs*ss,-1)).view(x_rand_conaug.size(0), x_rand_conaug.size(1), -1)
        x_rand_boneattaug = x_rand_boneattaug * self.temperature
        x_rand_boneattaug = self.softmax(x_rand_boneattaug)   
        boneaug = getbonelength(x_randaug2.detach().view(bs,ss,-1,3), self.boneindex)
        bonelengthaug = (boneaug * x_rand_boneattaug).sum(1)

        # Now deal with the bone direction prediction network

        #first sub-network
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        xscore = self.drop(self.relu(self.expand_bnscore(self.expand_convscore(xscore))))
        xbottom = [x.detach()]
        x = torch.cat((x, xscore*x),1)
        for i in range(len(self.pad) - 1):
            res = x[:, :self.channels, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
            xbottom.append(x.detach())
        x_1 = x
        bonedirect_1 = self.shrink_direct_1(x_1)
        bonedirect_1 = bonedirect_1.view(bonedirect_1.size(0),self.num_joints_out-1,3)
        bonesum_1 = torch.pow(torch.pow(bonedirect_1,2).sum(2),0.5).unsqueeze(2)
        bonedirect_1 = bonedirect_1/bonesum_1
        bonedirect_1 = bonedirect_1.view(bonedirect_1.size(0),(self.num_joints_out-1)*3,1)

        #second sub-network
        x = x.detach().repeat(1,1,int((self.sinrecfield*2+1)/self.filter_widths[0])).detach()
        for i in range(len(self.pad) - 1):
            x = torch.cat((x,xbottom[i]),1)
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            x = self.drop(self.relu(self.layers_bn1[2*i](self.layers_conv1[2*i](x))))
            x = res[:,:self.channels] + self.drop(self.relu(self.layers_bn1[2*i + 1](self.layers_conv1[2*i + 1](x))))
        bonedirect_2 = self.shrink_direct_2(x)
        bonedirect_2 = bonedirect_2.view(bonedirect_2.size(0),self.num_joints_out-1,3)
        bonesum_2 = torch.pow(torch.pow(bonedirect_2,2).sum(2),0.5).unsqueeze(2)
        bonedirect_2 = bonedirect_2/bonesum_2
        bonedirect_2 = bonedirect_2.view(bonedirect_2.size(0),(self.num_joints_out-1)*3,1)
        
        #compute the final 3D joint locations
        boned = self.directlinear(bonedirect_2.detach())
        bonel = self.lengthlinear(bonelength.detach())
        bonel = bonel.view(boned.size())
        x = bonel * boned
        x = self.shrink(x)
        x = x.view(x.size(0), -1, self.num_joints_out, 3)

        #compute the relative joint shifts independently based on the predictions of bl-network and each sub-network of bd-network
        bonejs_1 = self.lengthlinear_1(bonelength.detach())
        bonejs_2 = self.lengthlinear_2(bonelength.detach()) 

        js_1 = self.directlinear_1(bonedirect_1)
        bonejs_1 = bonejs_1.view(js_1.size())
        js_1 = js_1 * bonejs_1
        js_1 = self.shrink_js_1(js_1)
        js_1 = js_1.view(js_1.size(0), -1, self.jointshiftnum, 3)

        js_2 = self.directlinear_2(bonedirect_2)
        bonejs_2 = bonejs_2.view(js_2.size())
        js_2 = js_2 * bonejs_2
        js_2 = self.shrink_js_2(js_2)
        js_2 = js_2.view(js_2.size(0), -1, self.jointshiftnum, 3)
        return x, bonelength, x_rand, bonelengthaug, x_randaug, bonedirect_2, bonedirect_1, js_2, js_1



