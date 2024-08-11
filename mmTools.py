import sys
import os
import os.path as osp
import cv2
import numpy as np
from copy import deepcopy
import pickle
import pandas as pd
from tqdm import tqdm
from matching import *
import random
from functools import partial, lru_cache, wraps
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from itertools import product
from coco_ranges import * 
import torch

""" 
For the Snorkel stuff, we'll try to follow the same convention as 
https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb
"""

def predict_on_kp_seq (model, kpss, kpss_sc, tracked_seq) : 
    """ returns whether we are tracking the signer or not """
    from sign_detect_model import pad_to_len
    kp = [kpss[t][pos] for t, pos in tracked_seq]
    kp_sc = [kpss_sc[t][pos] for t, pos in tracked_seq]
    preds = [] 
    step_size = 5
    L = len(tracked_seq) 
    for i in range(0, L - 60 + 1, step_size) : 
        kpss_ = np.stack(kp[i:i+60])
        kpsc_ = np.stack(kp_sc[i:i+60])

        visible_mask = kpsc_ > 0.5
        vis_kps = kpss_[visible_mask]
        x, X, y, Y = vis_kps[..., 0].min(), vis_kps[..., 0].max(), vis_kps[..., 1].min(), vis_kps[..., 1].max()

        kpss_[..., 0] -= x + (X - x) / 2
        kpss_[..., 1] -= y + (Y - y) / 2

        S = max(X - x, Y - y)

        kpss_ /= (S / 2)

        kpss_[..., 1] = -kpss_[..., 1] # flip it
        kpss_[~visible_mask] = -1.0 # make things invisible

        kpss_ = torch.from_numpy(kpss_).squeeze(1).float()

        kpss_ = pad_to_len(kpss_, 60)

        batch = dict(
            pose_seq=kpss_.unsqueeze(0), 
            label=[0], 
            seq_hash=[''], 
        )

        pred = model.forward(batch, stage='val')['logits'].argmax().item()
        preds.append(pred)

    return mode(preds)


def mode (lst) :
    """ return the most frequently occuring value in list.
    Items must be hashable. In case of tie, return any
    """
    assert len(lst) > 0, "(mmTools:mode) Found an empty list"
    cnts = dict()
    for item in lst:
        if item not in cnts :
            cnts[item] = 1
        else :
            cnts[item] += 1
    mf = max(cnts.values())
    for k, v in cnts.items() :
        if v == mf:
            return k

def argmax(lst) :
    """
    Compute the argmax of an iterator
    """
    lst = list(lst)
    m = max(lst)
    return next(filter(lambda x : m == lst[x], range(len(lst))))

def are_hands_visible (keypoint_scores, threshold=0.5, *args, **kwargs) : 
    """ This is a strong constraint """
    hand_scores = keypoint_scores[:, coco_wholebody_hand_range]
    return np.median(hand_scores) >= threshold

def normalize_to_zero_one (keypoints, width, height) : 
    new_kp = np.copy(keypoints)
    new_kp[..., 0] /= width
    new_kp[..., 1] /= height
    return new_kp

def match_keypoint_score (kp1, kp2, kp_sc1, kp_sc2, width, height) : 
    kp1 = normalize_to_zero_one(kp1, width, height)
    kp2 = normalize_to_zero_one(kp2, width, height)
    # rel_ids = (coco_wholebody_body_range + coco_wholebody_face_range)
    rel_ids = (coco_wholebody_torso_and_face_range + coco_wholebody_face_range)
    rel_mask = np.zeros_like(kp_sc1, dtype=bool)
    rel_mask[..., rel_ids] = True
    both_vis = (kp_sc1 > 0.5) & (kp_sc2 > 0.5) & rel_mask
    if np.all(~both_vis) :
        return 1000.0 # large number
    score = np.abs(kp1[both_vis] - kp2[both_vis]).sum() / both_vis.sum()
    return score

def removeIndices (lst, indices, return_=False) :
    """
    In place removal of items at given indices. Obtained from :

        https://stackoverflow.com/questions/497426/deleting-multiple-elements-from-a-list
    """
    if return_ :
        stuff = []
        for i in sorted(indices, reverse=True) :
            stuff.append(lst[i])
            lst.remove(lst[i])
        return list(reversed(stuff))
    else :
        for i in sorted(indices, reverse=True):
            del lst[i]

def run_tracker (kpss, kpss_sc, terminate_threshold, width, height) : 
    T = len(kpss) 

    assert T > 0, 'I don\'t know what you want me to do with empty sequence' 

    alive_sequences = []
    dead_sequences = []

    for i, _ in enumerate(kpss[0]) : 
        alive_sequences.append([(0, i)]) # time and position in sequence

    ts = list(range(1, T))
    for t in tqdm(ts) : 
        kps_t, kps_sc_t = kpss[t], kpss_sc[t]
        n_alive = len(alive_sequences)

        # construct cost table
        cost_table = dict()
        for i in range(n_alive) : 
            tp, pos = alive_sequences[i][-1]
            for k in range(len(kps_t)) : 
                cost_table[(i, k)] = match_keypoint_score(
                    kpss[tp][pos], kps_t[k], 
                    kpss_sc[tp][pos], kps_sc_t[k],
                    width, height
                )

        match = optimalBipartiteMatching(cost_table) 

        unmatched_alive = [idx for idx in range(n_alive) if idx not in match]
        unmatched_new = [idx for idx in range(len(kps_t)) if idx not in match.values()]

        for i, k in match.items() : # same terminology
            if cost_table[(i, k)] < terminate_threshold :
                alive_sequences[i].append((t, k))
            else :
                unmatched_alive.append(i) 
                unmatched_new.append(k)

        removed = removeIndices(alive_sequences, unmatched_alive, return_=True)
        dead_sequences.extend(removed)

        for k in unmatched_new :
            alive_sequences.append([(t, k)])

    dead_sequences.extend(alive_sequences)
    return dead_sequences

def normalize_in_bounds (data): 
    """ normalize a tracked split represented as a dictionary in [-1, 1]^2 bounds """ 
    data = deepcopy(data)

    kpss = np.stack(data['kp'])
    kpsc = np.stack(data['kp_sc'])

    visible_mask = (kpsc > 0.5) & (~np.isnan(kpss).any(-1))
    vis_kps = kpss[visible_mask]
    x, X, y, Y = vis_kps[..., 0].min(), vis_kps[..., 0].max(), vis_kps[..., 1].min(), vis_kps[..., 1].max()
    
    kpss[..., 0] -= x + (X - x) / 2
    kpss[..., 1] -= y + (Y - y) / 2

    S = max(X - x, Y - y) 

    kpss /= (S / 2)

    kpss[..., 1] = -kpss[..., 1] # flip it
    kpss[~visible_mask] = -1.0 # make things invisible

    return kpss

def run_dwpose_on_video (model_ref, video_path) : 
    """ 
    run the dw pose model on the video path and additionally
    record timestamps for later use
    """
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened() : 
        success, image = cap.read()
        if not success: 
            break
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    # run dwpose
    output = list(tqdm(model_ref(video_path), total=len(timestamps)))

    for o, t in zip(output, timestamps): 
        o['timestamp'] = t

    return output
