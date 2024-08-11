import pandas as pd
import os
import os.path as osp
from tqdm import tqdm
import cv2
import pickle
from PIL import Image
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mmpose.apis import visualize
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.apis import MMPoseInferencer
import mmTools as mmt
import matplotlib
from itertools import cycle
from more_itertools import take
from sign_detect_model import load_model

def clamp (x, a, b) :
    return max(min(x, b), a)

def sample_k_colors (k) : 
    def unnorm_color (x) : 
        return tuple(map(lambda y : int(255 * y), x))
    hexes = list(take(k, cycle(matplotlib.colors.CSS4_COLORS.values())))
    norm_rgb = map(matplotlib.colors.to_rgb, hexes)
    unorm_rgb = list(map(unnorm_color, norm_rgb))
    return unorm_rgb

def pose_to_pose_data_sample (pose_dict) : 
    keypoints = pose_dict['keypoints']
    keypoint_score = pose_dict['keypoint_scores']

    tmp_instances = InstanceData()
    tmp_instances.keypoints = keypoints
    tmp_instances.keypoint_score = keypoint_score
    tmp_instances.keypoints_visible = keypoint_score

    tmp_datasample = PoseDataSample()
    tmp_datasample.pred_instances = tmp_instances

    return tmp_datasample

def images_to_video (video_path, images, metadata) : 
    videodims = images[0].shape[::-1][1:]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, metadata['framerate'], videodims)
    img = Image.new('RGB', videodims, color = 'darkred')
    for img in images :
        imtemp = img.copy()
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()

def imgArrayToPIL (arr) :
    """ utility to convert img array to PIL """
    if arr.dtype in [np.float32, np.float64, float] :
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype in [np.int32, np.int64, int]:
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType)

def aspectRatioPreservingResize (arr, smaller_dim) :
    """ utility for resizing image """
    pil_img = imgArrayToPIL(arr)
    h, w = pil_img.size
    if h < w :
        h, w = smaller_dim, smaller_dim * w / h
    else :
        h, w = smaller_dim * h / w, smaller_dim
    h, w = int(h), int(w)
    resized = pil_img.resize((h, w))
    np_arr = np.array(resized).astype(arr.dtype)
    if arr.dtype in [float, np.float32, np.float64] :
        np_arr /= 255.0
    return np_arr

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize poses')
    parser.add_argument('--save_to', type=str, default='vid.mp4', help='where to save')

    pose_pickle = '/home/ubuntu/general-purpose/ISL/dw_pose_clean/0ede63d064318ae6f7a6d17095164d67.pkl'
    pose_hash = '0ede63d064318ae6f7a6d17095164d67'

    args = parser.parse_args()

    model = load_model('lightning_logs/version_0')

    dwpose_model = MMPoseInferencer(
        pose2d='rtmpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py',
        pose2d_weights='rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth',
    )
    width, height = 1920, 1080

    with open(pose_pickle, 'rb') as fp : 
        pose_sequence = pickle.load(fp)
    
    st = random.randint(0, len(pose_sequence) - 1 - 500)
    en = st + 500

    print(f'Displaying from frame - {st} to {en}')

    video_path = '/home/ubuntu/general-purpose/ISL/videos/ish_news/zzJqSaaVIUU.webm'
    print('Path:', video_path)
    print('Hash:', pose_hash)

    visualizer = dwpose_model.inferencer.visualizer

    fig = plt.figure()
    cap = cv2.VideoCapture(video_path)

    # populate key points and key point scores
    kpss, kpss_sc = [], []
    
    for i in range(len(pose_sequence)) : 
        preds = pose_sequence[i]['predictions'] 
        keypoints, keypoint_scores = [], []

        assert len(preds) == 1, f'Oops, found predictions with length -- {len(preds)}' 
        for k in range(len(preds[0])) : 
            keypoints_np = np.array([preds[0][k]['keypoints']])
            keypoint_scores_np = np.array([preds[0][k]['keypoint_scores']])
            keypoints.append(keypoints_np)
            keypoint_scores.append(keypoint_scores_np)

        kpss.append(keypoints)
        kpss_sc.append(keypoint_scores)

    tracked_seqs = mmt.run_tracker(kpss, kpss_sc, 0.2, width, height)
    tracked_seqs = [_ for _ in tracked_seqs if len(_) >= 60]
    tracked_seqs = [_ for _ in tqdm(tracked_seqs) if mmt.predict_on_kp_seq(model, kpss, kpss_sc, _) == 1]
    
    colors = sample_k_colors(len(tracked_seqs))

    time_pos_to_seq_id = dict()
    for seq_id, seq in enumerate(tracked_seqs) :
        for item in seq: 
            time_pos_to_seq_id[item] = seq_id

    i = 0 
    images = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(kpss[i]) > 0 : 
            pose_dict=dict(
                keypoints=np.vstack(kpss[i]), 
                keypoint_scores=np.vstack(kpss_sc[i])
            )

            tmp_datasample = pose_to_pose_data_sample(pose_dict)
            visualizer.add_datasample(
                'visualization',
                image,
                tmp_datasample,
                skeleton_style='mmpose',
                draw_gt=False,
                kpt_thr=0.3
            )
            new_image = visualizer.get_image()
            for pos, kps in enumerate(kpss[i]): 
                if (i, pos) in time_pos_to_seq_id: 
                    color = colors[time_pos_to_seq_id[(i, pos)]]
                    mask = kpss_sc[i][pos] > 0.5
                    x, y, X, Y = kps[mask, 0].min(), kps[mask, 1].min(), kps[mask, 0].max(), kps[mask, 1].max()
                    x = clamp(x, 0, width)
                    X = clamp(X, 0, width)
                    y = clamp(y, 0, height)
                    Y = clamp(Y, 0, height)
                    cv2.rectangle(new_image, (int(x), int(y)), (int(X), int(Y)), color, 2)
                    
        else :
            new_image = image

        if i >= st : 
            images.append(aspectRatioPreservingResize(new_image, 512))

        if i >= en : 
            break

        i += 1

    cap.release()
    images_to_video(args.save_to, images, dict(framerate=29))
