# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from common.human import *

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def pck(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    #print(dis.size())
    t = torch.Tensor([0.15]).cuda()  # threshold
    out = (dis < t).float() * 1
    return out.sum()/14.0

def auc(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    outall = 0
    #print(dis.size())
    for i in range(150):
        t = torch.Tensor([float(i)/1000]).cuda()  # threshold
        out = (dis < t).float() * 1
        outall+=out.sum()/14.0
    outall = outall/150
    return outall

    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))


def joint_collision(predicted, target, weight, thres=0.1):
    """
    verify whether predicted and target joints lie within a given threshold
    if True -> collision -> punish

    :return: a weight matrix of shape (bs, 17)
    """
    diff = torch.linalg.norm(predicted - target, dim=2) < thres
    diff = diff.double() + 1
    weight *= diff

    return weight


def is_so(M):
    det = cmath.isclose(torch.linalg.det(M), 1, rel_tol=1e-03)
    orth = cmath.isclose(torch.linalg.det(M@M.T), 1, rel_tol=1e-03)
    return 1 if orth and det else 2


def maev(predicted, target):
    """
    MAEV: Mean Absolute Error of Vectors
    :param predicted: (bs,16,9) tensor
    :param target:  (bs,16,9) tensor
    average error of 16 bones
    """
    bs, num_bones = predicted.shape[0], predicted.shape[1]
    predicted = predicted.view(bs,num_bones,3,3)
    target = target.view(bs,num_bones,3,3)
    w_arr = torch.ones(target.shape[:2])
    for b in range(bs):
        for bone in range(num_bones):
            M = predicted[b,bone]
            w_arr[b,bone] = is_so(M)
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
        w_arr = w_arr.cuda()
    aev = torch.norm(torch.norm(predicted - target, dim=len(target.shape)-2), dim=len(target.shape)-2)
    maev = torch.mean(aev*w_arr)
    return maev


# 2. L2 norm on unit bone vectors

def mbve(predicted, target):
    """
    MBVE - Mean Bone Vector Error
    """
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
    bs, num_bones = predicted.shape[0], predicted.shape[1]

    pred_info = torch.zeros(bs, num_bones, 3)
    tar_info = torch.zeros(bs, num_bones, 3)

    pred = Human(1.8, "cpu")
    pred_model = pred.update_pose(predicted)
    tar = Human(1.8, "cpu")
    tar_model = tar.update_pose(target)
    for b in range(bs):
        pred_info[b,:] = vectorize(pred_model)[:,:3]
        tar_info[b,:] = vectorize(tar_model)[:,:3]
    mbve = torch.norm(pred_info - tar_info)
    return mbve


# 3. Decompose SO(3) into Euler angles

def meae(predicted, target):
    """
    MEAE: Mean Euler Angle Error
    :param predicted: a (1,16,9) tensor
    :param target: a (1,16,9) tensor
    e.g. Decomposing a yields = (0,0,45) deg = (0,0,0.7854) rad
    sum of 3 ele is 0.7854, avg of 16 bones is 0.7854
    """
    if torch.cuda.is_available():
        predicted = predicted.cuda()
        target = target.cuda()
    bs, num_bones = predicted.shape[0], predicted.shape[1]
    predicted = predicted.view(bs,num_bones,3,3)
    target = target.view(bs,num_bones,3,3)

    pred_euler = torch.zeros(bs,num_bones,3)
    tar_euler = torch.zeros(bs,num_bones,3)
    for b in range(bs):
        for bone in range(num_bones):
            pred_euler[b,bone,:] = torch.tensor(rot_to_euler(predicted[b,bone]))
            tar_euler[b,bone,:] = torch.tensor(rot_to_euler(target[b,bone]))
    return torch.mean(torch.sum(pred_euler - tar_euler, dim=2))

