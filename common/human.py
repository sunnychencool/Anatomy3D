import numpy as np
import cmath
import torch


def rot(euler: tuple) -> torch.tensor:
    """
    General rotation matrix
    :param euler: (a, b, r) rotation in rad in ZYX
    
    :return R: a rotation matrix R
    """
    from math import sin, cos
    a, b, r = euler[0], euler[1], euler[2]
    row1 = torch.tensor([cos(a)*cos(b), cos(a)*sin(b)*sin(r)-sin(a)*cos(r), cos(a)*sin(b)*cos(r)+sin(a)*sin(r)])
    row2 = torch.tensor([sin(a)*cos(b), sin(a)*sin(b)*sin(r)+cos(a)*cos(r), sin(a)*sin(b)*cos(r)-cos(a)*sin(r)])
    row3 = torch.tensor([-sin(b), cos(b)*sin(r), cos(b)*cos(r)])
    R = torch.stack((row1, row2, row3), 0)
    assert cmath.isclose(torch.linalg.det(R), 1, rel_tol=1e-04), torch.linalg.det(R)
    return R


def rot_to_euler(R: np.array) -> np.array:
    """
    :return: Euler angles in rad in ZYX
    """
    import cv2 as cv
    if torch.is_tensor(R):
        R = R.detach().cpu().numpy()
    angles = np.radians(cv.RQDecomp3x3(R)[0])
    angles[0], angles[2] = angles[2], angles[0]
    return angles


class Human:
    """ Implementation of Winter human model """
    def __init__(self, H, device="cuda:0"):
        self.device = device
        self.half_face = 0.066*H
        self.neck = 0.052*H
        self.upper_spine, self.lower_spine = 0.144*H, 0.144*H
        self.clavicle = 0.129*H
        self.upper_arm, self.lower_arm = 0.186*H, 0.146*H

        self.pelvis = 0.191*H
        self.thigh, self.calf = 0.245*H, 0.246*H
        self.root = torch.zeros(3, device=self.device)

        self.child = {
            # child : parent
            "upper_spine": "lower_spine",
            "head": "neck",
            "l_lower_arm": "l_upper_arm",
            "r_lower_arm": "r_upper_arm",
            "l_calf": "l_thigh",
            "r_calf": "r_thigh",
        }

        self.constraints = {
            "lower_spine": ((-0.61,0.61), (-0.52,0.52), (-0.52,1.31)),
            "upper_spine": ((0,0), (0,0), (0,1.66)),
            "neck": ((0,0), (0,0), (0,1.22)),
            "head": ((-0.61,0.61), (-1.22,1.22), (-0.96,1.39)),

            "l_clavicle": ((0,0), (0,0), (0,0)), #4
            "l_upper_arm": ((-1.57,2.28), (-0.707,2.27), (-1.57,3.14)),
            "l_lower_arm": ((0,0), (0,2.62), (0,0)),
            "r_clavicle": ((0,0), (0,0), (0,0)),
            "r_upper_arm": ((-2.28,1.57), (-2.27,0.707), (-1.57,3.14)),
            "r_lower_arm": ((0,0), (-2.62,0), (0,0)),

            "l_hip": ((0,0), (0,0), (0,0)), #10
            "l_thigh": ((-0.87,0.35), (-0.785,0.785), (-2.09,0.52)),
            "l_calf": ((0,0), (0,0), (0,2.79)),
            "r_hip": ((0,0), (0,0), (0,0)),
            "r_thigh": ((-0.35,0.87), (-0.785,0.785), (-2.09,0.52)),
            "r_calf": ((0,0), (0,0), (0,2.79)),
        }


    def init_bones(self):
        self.bones = {
            "lower_spine": torch.tensor([0, -self.lower_spine, 0]),
            "upper_spine": torch.tensor([0, -self.upper_spine, 0]),
            "neck": torch.tensor([0, -self.neck, 0]),
            "head": torch.tensor([0, -self.half_face, 0]),

            "l_clavicle": torch.tensor([self.clavicle, 0, 0]),
            "l_upper_arm": torch.tensor([self.upper_arm, 0, 0]),
            "l_lower_arm": torch.tensor([self.lower_arm, 0, 0]),
            "r_clavicle": torch.tensor([-self.clavicle, 0, 0]),
            "r_upper_arm": torch.tensor([-self.upper_arm, 0, 0]),
            "r_lower_arm": torch.tensor([-self.lower_arm, 0, 0]),

            "l_hip": torch.tensor([self.pelvis/2, 0, 0]),
            "l_thigh": torch.tensor([0, self.thigh, 0]),
            "l_calf": torch.tensor([0, self.calf, 0]),
            "r_hip": torch.tensor([-self.pelvis/2, 0, 0]),
            "r_thigh": torch.tensor([0, self.thigh, 0]),
            "r_calf": torch.tensor([0, self.calf, 0])
        }
        self.bones = { bone: self.bones[bone].to(self.device) for bone in self.bones.keys() }
        

    def check_range(self, bone, angles):
        punish_w = 1.0
        for i in range(3):
            low = self.constraints[bone][i][0]
            high = self.constraints[bone][i][1]
            if high != low:
                if angles[i] < low:
                    angles[i] = low
                    punish_w += 1.0
                elif angles[i] > high:
                    angles[i] = high
                    punish_w += 1.0
        return angles, punish_w


    def check_constraints(self, bone, R: np.array, parent=None):
        """
        Punish (by adding weights) if NN outputs are beyond joint rotation constraints.
        """
        import torch.nn.functional as f
        absolute_angles = rot_to_euler(np.array(R).reshape(3,-1))
        if parent is not None:
            parent_angles = rot_to_euler(parent.detach().cpu().numpy())
            child_angles = absolute_angles
            relative_angles = child_angles - parent_angles
            aug_angles, punish_w = self.check_range(bone, relative_angles)
            R = rot(aug_angles + parent_angles)
        else:
            aug_angles, punish_w = self.check_range(bone, absolute_angles)
            R = rot(aug_angles)
        return f.normalize(R.to(torch.float32).to(self.device)), punish_w


    def sort_rot(self, elem):
        """
        :param ang: a list of 144 elements (9 * 16)
        process NN output to rotation matrix of 16 bones
        """
        elem = elem.flatten()
        self.rot_mat, self.punish_list = {}, []
        for k, bone in enumerate(self.constraints.keys()):
            R = elem[9*k:9*(k+1)].detach().numpy()
            if bone in self.child.keys():
                parent = self.child[bone]
                self.rot_mat[bone], punish_w = self.check_constraints(bone, R, self.rot_mat[parent])
            else:
                self.rot_mat[bone], punish_w = self.check_constraints(bone, R)
            self.punish_list.append(punish_w)


    def update_bones(self, elem=None):
        """
        Initiates a T-Pose human model and rotate each bone using the given rotation matrices
        :return model: a numpy array of (17,3)
        """
        self.init_bones()
        if elem is not None:
            self.sort_rot(elem)
            self.bones = { bone: self.rot_mat[bone] @ self.bones[bone] for bone in self.constraints.keys() }


    def update_pose(self, elem=None):
        """
        Assemble bones to make a human body
        """
        self.update_bones(elem)

        root = self.root
        lower_spine = self.bones["lower_spine"]
        neck = self.bones["upper_spine"] + lower_spine
        chin = self.bones["neck"] + neck
        nose = self.bones["head"] + chin

        l_shoulder = self.bones["l_clavicle"] + neck
        l_elbow = self.bones["l_upper_arm"] + l_shoulder
        l_wrist = self.bones["l_lower_arm"] + l_elbow
        r_shoulder = self.bones["r_clavicle"] + neck
        r_elbow = self.bones["r_upper_arm"] + r_shoulder
        r_wrist = self.bones["r_lower_arm"] + r_elbow

        l_hip = self.bones["l_hip"]
        l_knee = self.bones["l_thigh"] + l_hip
        l_ankle = self.bones["l_calf"] + l_knee
        r_hip = self.bones["r_hip"]
        r_knee = self.bones["r_thigh"] + r_hip
        r_ankle = self.bones["r_calf"] + r_knee

        self.model = torch.stack((neck, lower_spine, root, chin, nose,
                l_shoulder, l_elbow, l_wrist, r_shoulder, r_elbow, r_wrist,
                l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle), 0)
        return self.model


def vectorize(gt_3d, dataset="mpi") -> torch.tensor:
    """
    process gt_3d (17,3) into a (16,4) that contains bone vector and length
    :return bone_info: [unit bone vector (,3) + bone length (,1)]
    """
    if dataset == "mpi":
        indices = (
            (2,1), (1,0), (0,3), (3,4),  # spine + head
            (0,5), (5,6), (6,7), 
            (0,8), (8,9), (9,10), # arms
            (2,11), (11,12), (12,13),
            (2,14), (14,15), (15,16), # legs
        )
    elif dataset == "h36m":
        indices = (
            (0,7), (7,8), (8,9), (9,10),  # spine + head
            (8,11), (11,12), (12,13), 
            (8,14), (14,15), (15,16), # arms
            (0,4), (4,5), (5,6),
            (0,1), (1,2), (2,3), # legs
        )
    else:
        print("Unrecognized dataset name.")

    num_bones = len(indices)
    try:
        gt_3d_tensor = torch.from_numpy(gt_3d)
    except TypeError:
        gt_3d_tensor = gt_3d

    bone_info = torch.zeros([num_bones, 4], requires_grad=False) # (16, 4)
    for i in range(num_bones):
        vec = gt_3d_tensor[indices[i][1],:] - gt_3d_tensor[indices[i][0],:]
        vec_len = torch.linalg.norm(vec)
        unit_vec = vec/vec_len
        bone_info[i,:3], bone_info[i,3] = unit_vec, vec_len
    return bone_info
