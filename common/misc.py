import torch
import numpy as np
from common.human import *


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_rot_from_vecs(vec1: np.array, vec2: np.array) -> np.array:
    """ 
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector

    :return R: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    
    Such that vec2 = R @ vec1
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R


def convert_gt(gt_3d: np.array, t_info, dataset="mpi") -> np.array:
    """
    Compare GT3D kpts with T pose and obtain 16 rotation matrices

    :return R_stack: a (16,9) arrays with flattened rotation matrix for 16 bones
    """
    # process GT
    bone_info = vectorize(gt_3d, dataset)[:,:3] # (16,3) bone vecs

    num_row = bone_info.shape[0]
    R_stack = np.zeros([num_row, 9])
    # get rotation matrix for each bone
    for k in range(num_row):
        R = get_rot_from_vecs(t_info[k,:], bone_info[k,:]).flatten()
        R_stack[k,:] = R
    return R_stack
