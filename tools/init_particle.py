#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
from fcn.run import run_network
from fcn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import tensorflow as tf
import os.path as osp
import numpy as np
import cv2

from pathlib import Path

def init_particle(images, meta_data = None):

    root            = os.path.join(osp.dirname(__file__), "..")
    cfg_file        = os.path.join(root, "experiments/cfgs/lov_color_2d.yml")
    imdb_name       = "lov_keyframe"
    gpu_id          = 0
    rig_name        = os.path.join(root, "data/LOV/camera.json")
    cad_name        = os.path.join(root, "data/LOV/models.txt")
    pose_name       = os.path.join(root, "data/LOV/poses.txt")
    background_name = os.path.join(root, "data/cache/backgrounds.pkl")
    network_name    = "vgg16_convs"
    model           = os.path.join(root, "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt")

    if cfg_file is not None:
        cfg_from_file(cfg_file)

    # print('Using config:')
    # pprint.pprint(cfg)

    imdb = get_imdb(imdb_name)

    # construct meta data
    if meta_data is None:
        K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
        meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 10000.0})
    # print(meta_data)

    cfg.GPU_ID = gpu_id
    device_name = '/gpu:{:d}'.format(gpu_id)
    # print(device_name)

    cfg.TRAIN.NUM_STEPS = 1
    cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
    cfg.TRAIN.TRAINABLE = False

    cfg.RIG = rig_name
    cfg.CAD = cad_name
    cfg.POSE = pose_name
    cfg.BACKGROUND = background_name
    cfg.IS_TRAIN = False

    from networks.factory import get_network
    network = get_network(network_name)
    print('Use network `{:s}` in training'.format(network_name))

    # start a session
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    saver.restore(sess, model)
    print(('Loading model weights from {:s}').format(model))

    return run_network(sess, network, imdb, images, meta_data)

if __name__ == "__main__":

    # construct the images
    root = 'data/demo_images/'
    num = 5
    images = []
    for i in range(num):
        rgb_filename = root + '{:06d}-color.png'.format(i+1)
        rgb_im = cv2.imread(rgb_filename, cv2.IMREAD_UNCHANGED)
        depth_filename = root + '{:06d}-depth.png'.format(i+1)
        depth_im = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
        images.append((rgb_im, depth_im))

    out = init_particle(images)
