#!/usr/bin/env python3

# mmposeのチュートリアルより
# https://mmpose.readthedocs.io/en/latest/user_guides/inference.html
# https://github.com/open-mmlab/mmpose/blob/dev-1.x/demo/topdown_demo_with_mmdet.py

from argparse import ArgumentParser
import os
import subprocess

import cv2
import numpy as np
import torch

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    # MMPoseInferencer,
    inference_topdown,
    init_model as init_pose_estimator,
)
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("input")
args = parser.parse_args()

image = cv2.imread(args.input)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# inferencer = MMPoseInferencer('wholebody')
# result_generator = inferencer(rgb_image, show=True)
# result = next(result_generator)
# print(result)

det_model_cfg_name = "rtmdet_m_8xb32-300e_coco"
det_model_cfg = "config/" + det_model_cfg_name + ".py"
det_ckpt = "config/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
if not os.path.exists(det_model_cfg):
    subprocess.run(["mim", "download", "mmdet", "--config", det_model_cfg_name, "--dest", "config"])

pose_model_cfg_name = "rtmpose-m_8xb64-270e_coco-wholebody-256x192"
pose_model_cfg = "config/" + pose_model_cfg_name + ".py"
pose_ckpt = "config/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
# pose_model_cfg_name = "rtmpose-m_8xb256-420e_coco-256x192"
# pose_model_cfg = "config/" + pose_model_cfg_name + ".py"
# pose_ckpt = "config/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
if not os.path.exists(pose_model_cfg):
    subprocess.run(["mim", "download", "mmpose", "--config", pose_model_cfg_name, "--dest", "config"])

# mmdet用のパラメータ
cat_id = 0  # COCO
bbox_thr = 0.3
nms_thr = 0.3
kpt_thr = 0.3

# init model
detector = init_detector(det_model_cfg, det_ckpt, device=device)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)
pose_estimator = init_pose_estimator(pose_model_cfg, pose_ckpt, device=device)
# build the visualizer
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# set skeleton, colormap and joint connection rule
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

# inference on a single image
det_result = inference_detector(detector, rgb_image)
pred_instance = det_result.pred_instances.cpu().numpy()
bboxes = np.concatenate(
    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == cat_id,
                               pred_instance.scores > bbox_thr)]
bboxes = bboxes[nms(bboxes, nms_thr), :4]
pose_results = inference_topdown(pose_estimator, rgb_image, bboxes)
# merge results as a single data sample
results = merge_data_samples(pose_results)
# visualize the results
output_image = visualizer.add_datasample(
    'result',
    rgb_image,
    data_sample=results,
    draw_bbox=True,
    #show=True
)

output_bgr_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
cv2.imshow("result", output_bgr_image)
cv2.waitKey()
print(results.get('pred_instances', None))
