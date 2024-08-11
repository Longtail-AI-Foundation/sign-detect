"""
This file is read-only and records:

    * The heuristics we used for detecting hand signers
    * How we used them in conjunction with snorkel

We don't expect this code to work __as_is__ because a lot
of intermediate files are not part of this repository. 

Anyway, here is a high level overview: 

    * In a preprocessing step, we extracted human keypoints from news footage
    * We created a small dev set where we classified around 400 keypoint sequences. 
    * We developed 7 heuristic functions that label these keypoint sequences.
    * We fit a label model (using snorkel) that assigns probabilistic labels to keypoints

Later, we use the probabilistic labels to train a neural net. The neural net has better
error rate (measured on our dev set) and is easier to distribute.
""" 
import sys
import os
import os.path as osp
import numpy as np
from copy import deepcopy
import pickle
import isl_utils as iu
import pandas as pd
from tqdm import tqdm
import random
from functools import partial, lru_cache, wraps
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from itertools import product
import torch
from coco_ranges import * 


""" 
For the Snorkel stuff, we'll try to follow the same convention as 
https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb
"""

SNORKEL_ENABLE = False

ABSTAIN = -1

NOT_SIGNER = 0
SIGNER = 1
LABELS = ['Abstain', 'Doesn\'t Contain SL', 'Contains SL']

def snorkel_wrapper(enable) : 
    """
    This decorator interprets the input as a row (pandas) or a dictionary depending
    on whether it is enabled. If enabled, also apply the snorkel decorator

    All heuristics have this decorator applied on them
    """
    def wrapper(func):
        @wraps(func)
        def inner(rowOrData, **kwargs):
            if enable : 
                # interpret arg as row
                data = load_split_from_hash(rowOrData.hash)
                return func(data, **kwargs)
            else : 
                # interpret arg as data
                return func(rowOrData, **kwargs)
        if enable : 
            return labeling_function()(inner)
        else :
            return inner
    return wrapper

@snorkel_wrapper(SNORKEL_ENABLE)
def num_ongoing_track_heuristic(data) :
    """
    In most cases, there is only one signer per video so if there are many individuals 
    are detected, then the probability of anyone of them being the signer is low.

    From this you can get a sense of how crude some of our heuristcs are. 
    """
    u = random.random() 
    if u < (data['num_ongoing_tracks'] - 1) / (data['num_ongoing_tracks']) :
        return NOT_SIGNER
    else :
        return SIGNER

@snorkel_wrapper(SNORKEL_ENABLE)
def video_path_heuristic (data) : 
    """ 
    Since our complete dataset also includes ISL dictionaries, if the individual is coming
    from that dataset, we can be fairly certain that they are a signer.
    """
    path = data['video_path']
    pref_path = osp.split(path)[0]
    sign_ds = ['alphabet', 'includeby2', 'isl-scenarios', 'islrtc', 'rkm', 'split_rkm'] 
    if any(_ in pref_path for _ in sign_ds) : 
        return SIGNER
    else :
        return ABSTAIN

@snorkel_wrapper(SNORKEL_ENABLE)
def legs_visible_heuristic (data, threshold=0.2) : 
    """
    In the news videos, the signer is often sitting down or standing close to the camera
    so that their legs are occluded/not visible. We use this information to heuristically 
    decide whether someone is a hand-signer.
    """
    all_scores  = np.stack(data['kp_sc'])
    leg_scores = all_scores[..., coco_wholebody_legs_range]
    if np.median(leg_scores) >= threshold: 
        return NOT_SIGNER
    else :
        return SIGNER

@snorkel_wrapper(SNORKEL_ENABLE)
def only_one_person_heuristic (data) : 
    """ 
    Since we mined only sign language videos, we are hopeful that if one person 
    is detected, they are the hand signer
    """
    if data['num_ongoing_tracks'] == 1 :
        return SIGNER
    else :
        return ABSTAIN

@snorkel_wrapper(SNORKEL_ENABLE)
def bounding_box_heuristic (data) : 
    """
    If the person detected occupies a very small portion of the screen, then they
    are unlikely to be a hand-signer
    """
    kpss = np.stack(data['kp'])
    kpsc = np.stack(data['kp_sc']) 
    visible_mask = kpsc > 0.5
    vis_kps = kpss[visible_mask]
    x, X, y, Y = vis_kps[..., 0].min(), vis_kps[..., 0].max(), vis_kps[..., 1].min(), vis_kps[..., 1].max()
    metadata = iu.get_metadata_by_hash(data['pose_hash'])
    width, height = metadata['width'], metadata['height'] 
    if (X - x) / width < 0.1 or (Y - y) / height < 0.1 : 
        return NOT_SIGNER
    else: 
        return SIGNER

@snorkel_wrapper(SNORKEL_ENABLE)
def movement_heuristic (data) : 
    """
    If a person's hands are moving a lot, they are probably a hand-signer. This is
    our single most powerful heuristic.
    """
    kpss = np.stack(data['kp'])
    kpsc = np.stack(data['kp_sc']) 

    # first compute the bounds
    visible_mask = kpsc > 0.5
    vis_kps = kpss[visible_mask]
    x, X, y, Y = vis_kps[..., 0].min(), vis_kps[..., 0].max(), vis_kps[..., 1].min(), vis_kps[..., 1].max()

    diffs = []
    skip_rate = 10
    for i in range(kpss.shape[0] - skip_rate) : 
        rel_mask = np.zeros_like(kpsc[i], dtype=bool)
        rel_mask[..., coco_wholebody_hand_range] = True
        mask = (kpsc[i] > 0.5) & (kpsc[i + skip_rate] > 0.5) & rel_mask
        dxdy = (kpss[i, mask] - kpss[i + skip_rate, mask]) / max(X - x, Y - y)
        l1_dxdy = np.abs(dxdy).sum(axis=1)
        avg_l1 = np.mean(l1_dxdy)
        diffs.append(avg_l1)

    if np.median(diffs) > 0.1 : 
        return SIGNER
    else : 
        return NOT_SIGNER

@snorkel_wrapper(SNORKEL_ENABLE)
def chest_level_heuristic (data, chest_frac=0.0, cross_count=0, cross_frac=0.1) : 
    """ 
    if the hands cross the chest level often, it is likely that the 
    person is doing sign language
    """
    kpss = np.stack(data['kp'])
    kpsc = np.stack(data['kp_sc']) 

    # first compute the bounds
    visible_mask = kpsc > 0.5
    vis_kps = kpss[visible_mask]
    x, X, y, Y = vis_kps[..., 0].min(), vis_kps[..., 0].max(), vis_kps[..., 1].min(), vis_kps[..., 1].max()

    cross_chest_level = 0 
    for i in range(kpss.shape[0]) : 
        shoulder_y = kpss[i, 0, coco_wholebody_shoulder_range, 1].mean()
        chest_y = shoulder_y + chest_frac * (Y - y) # y axis increases downwards

        rel_mask = np.zeros_like(kpsc[i], dtype=bool)
        rel_mask[..., coco_wholebody_hand_range] = True
        mask = (kpsc[i] > 0.5) & rel_mask

        if (kpss[i, mask][..., 1] < chest_y).sum() > cross_count : # y axis increases downwards
            cross_chest_level += 1 

    if cross_chest_level > (cross_frac * kpss.shape[0]):
        return SIGNER
    else : 
        return NOT_SIGNER

def snorkel_label_model_fitting() : 
    """
    This code section was responsible for fitting a label model for 
    tracking the hand-signer in videos.
    """
    global SNORKEL_ENABLE
    SNORKEL_ENABLE = True
    iu.seed_everything(0)

    # these files are not distributed with this repository.The csvs just contain a list of 
    # identifiers for human keypoint sequences that are to be labelled by our heuristics
    unlabeled_set = pd.read_csv(osp.join(iu.METADATA_DIR, 'unlabelled_track_splits_1M.csv'))
    full_set = pd.read_csv(osp.join(iu.METADATA_DIR, 'tracked_splits_class.csv'))
    full_set = full_set.sample(frac=1).reset_index(drop=True)
    dev_set = full_set.head(400)
    test_set = full_set[400:]

    lfs = [
        num_ongoing_track_heuristic, 
        legs_visible_heuristic, 
        only_one_person_heuristic, 
        bounding_box_heuristic, 
        movement_heuristic, 
        chest_level_heuristic, 
        video_path_heuristic
    ]

    applier = PandasLFApplier(lfs=lfs)
    # this step takes a bit of time, on the order of a few hours.
    L_train = applier.apply(df=unlabeled_set)

    label_model = LabelModel(cardinality=2, verbose=True)
    # this fitting step should take < 10 min
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
