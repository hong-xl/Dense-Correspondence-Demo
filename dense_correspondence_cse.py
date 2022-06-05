#!/usr/bin/env python3
# Copied from
# https://github.com/facebookresearch/detectron2/blob/bb96d0b01d0605761ca182d0e3fac6ead8d8df6e/projects/DensePose/apply_net.py
# with some simplification

import cv2
import numpy as np
import argparse
import logging
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.densepose_outputs_vertex import (
    get_texture_atlases,
)

from densepose.vis.densepose_results_textures import (
    get_texture_atlas,
)

from utils import DensePoseOutputsVertexVisualizer, create_extractor


# this code only works perfrctly for human's cse
VISUALIZERS = {
    "dp_vertex": DensePoseOutputsVertexVisualizer,
}

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default='configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml', metavar="<config>", help="Config file")
parser.add_argument("--model", default='R_50_FPN_DL_s1x.pkl', metavar="<model>", help="Model file")
parser.add_argument("--input", default='images/image.jpg', metavar="<input>", help="Input data")
parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER, )
parser.add_argument("--visualizations", default='dp_vertex', metavar="<visualizations>", help="Comma separated list of visualizations, possible values: "
                                                                         "[{}]".format(",".join(sorted(VISUALIZERS.keys()))),)
parser.add_argument("--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold")
parser.add_argument("--min_score", metavar="<score>", default=0.8, type=float, help="Minimum detection score to visualize",)
parser.add_argument("--texture_atlas", metavar="<texture_atlas>", default=None, help="Texture atlas file (for IUV texture transfer)", )
parser.add_argument("--texture_atlases_map", metavar="<texture_atlases_map>", default=None, help="JSON string of a dict containing texture atlas files for each mesh",)
parser.add_argument("--output", metavar="<image_file>", default="output.png", help="File name to save output to", )
parser.add_argument("--device", metavar="<image_file>", default="cpu", help="Device to inference", )
args = parser.parse_args()

mode = 'human'  # animal or human
if mode == 'human':
    args.input = 'images/image.jpeg'
    args.cfg = 'configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml'
    args.model = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_DL_s1x/251156349/model_final_e96218.pkl'

else:
    args.input = 'images/sheep.jpg'
    args.cfg = 'configs/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_16k.yaml'
    args.model = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_16k/270727112/model_final_421d28.pkl'

logger = logging.getLogger()
opts = []
opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
opts.append(str(args.min_score))
if args.nms_thresh is not None:
    opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
    opts.append(str(args.nms_thresh))

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(args.cfg)
cfg.merge_from_list(args.opts)
if opts:
    cfg.merge_from_list(opts)
cfg.MODEL.WEIGHTS = args.model
cfg.MODEL.DEVICE = args.device
cfg.freeze()
vis_spec = args.visualizations


predictor = DefaultPredictor(cfg)

texture_atlas = get_texture_atlas(args.texture_atlas)
texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
visualizer = VISUALIZERS[vis_spec](
        cfg=cfg,
        inplace=True,
        device=args.device,
        texture_atlas=texture_atlas,
        texture_atlases_dict=texture_atlases_dict,
    )

extractor = create_extractor(visualizer)


img = read_image(args.input, format="BGR")  # predictor expects BGR image.
with torch.no_grad():
    outputs = predictor(img)["instances"]
logger.info(f"Processing {args.input}")
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
data = extractor(outputs)
img = img if mode == 'animal' else image
image_vis = visualizer.visualize(img, data)
cv2.imwrite(args.output, image_vis)



