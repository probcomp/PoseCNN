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
import pyrealsense2 as rs

from pathlib import Path

def init_particle(images, meta_data = None, bag_file = None):
    """ Provides a trace particle initialization from image data.

    Parameters
    ----------
    images : [(str, str)] or [(np.ndarray, np.ndarray)]
        Set of image paths or images to run through the network.
    meta_deta : dict(str => any)
        Optional. Must contain camera intrinsics in 'intrinsic_matrix',
        and depth camera factor in 'factor_depth'.
    bag_file : str
        Optional. Specifies location of bag with appropriate camera metadata
        (typically the recorded bag from which the images were taken).

    Returns
    -------
    [[(str, np.ndarray)]]
        List of detections from each scene, where detections are a pair
        of the object name to its detected 6D pose.
    """

    # Check if we've got image paths instead of images.
    if isinstance(images[0][0], str):
        loaded_images = []
        for (rgb_filename, depth_filename) in images:
            rgb_im = cv2.imread(rgb_filename, cv2.IMREAD_UNCHANGED)
            depth_im = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
            loaded_images.append((rgb_im, depth_im))
        images = loaded_images

    root            = os.path.join(osp.dirname(__file__), "..")
    cfg_file        = os.path.join(root, "experiments/cfgs/obsd.yml")
    if bag_file is None:
        bag_file        = "/om2/user/agarret7/projects/online-bayesian-scene-derendering/data/mvd0/record.bag"
    imdb_name       = "lov_keyframe"
    gpu_id          = 0
    # rig_name        = os.path.join(root, "data/LOV/camera.json")
    cad_name        = os.path.join(root, "data/LOV/models.txt")
    pose_name       = os.path.join(root, "data/LOV/poses.txt")
    background_name = os.path.join(root, "data/cache/backgrounds.pkl")
    network_name    = "vgg16_convs"
    model           = os.path.join(root, "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt")

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)

    ctx = rs.context()
    d = ctx.load_device(bag_file)
    depth_sensor, color_sensor = d.query_sensors()
    color_prof, depth_prof = color_sensor.get_stream_profiles()[0], depth_sensor.get_stream_profiles()[0]

    config.enable_stream(rs.stream.color, color_prof.format(), color_prof.fps())
    config.enable_stream(rs.stream.depth, depth_prof.format(), depth_prof.fps())

    # Start streaming
    profile = pipeline.start(config)

    rgb_profile = profile.get_stream(rs.stream.color)
    intr = rgb_profile.as_video_stream_profile().get_intrinsics()

    depth_sensor = profile.get_device().first_depth_sensor()
    factor_depth = int(1. / depth_sensor.get_depth_scale())

    pipeline.stop()

    if cfg_file is not None:
        cfg_from_file(cfg_file)

    # print('Using config:')
    # pprint.pprint(cfg)

    imdb = get_imdb(imdb_name)

    # construct meta data
    if meta_data is None:
        # fx = 1066.778
        # fy = 1067.487
        # px = 312.9869
        # py = 241.3109
        # factor_depth = 10000.0

        fx = intr.fx
        fy = intr.fy
        px = intr.ppx
        py = intr.ppy

        K = np.array([[fx,  0, px],
                      [ 0, fy, py],
                      [ 0,  0,  1]])

        meta_data = dict({'intrinsic_matrix': K, 'factor_depth': factor_depth})

    cfg.GPU_ID = gpu_id
    device_name = '/gpu:{:d}'.format(gpu_id)
    # print(device_name)

    cfg.TRAIN.NUM_STEPS = 1
    cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
    cfg.TRAIN.TRAINABLE = False

    # cfg.RIG = rig_name
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
    # root = 'data/demo_images/'
    root = "/om2/user/agarret7/projects/online-bayesian-scene-derendering/data/mvd0/"
    num = 5
    images = []

    # for i in range(num):
    #     rgb_filename = root + '{:06d}-color.png'.format(i+1)
    #     rgb_im = cv2.imread(rgb_filename, cv2.IMREAD_UNCHANGED)
    #     depth_filename = root + '{:06d}-depth.png'.format(i+1)
    #     depth_im = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
    #     images.append((rgb_im, depth_im))

    rgb_im = cv2.imread(os.path.join(root, "img", "adjacent_color.png"), cv2.IMREAD_UNCHANGED)
    depth_im = cv2.imread(os.path.join(root, "img", "adjacent_depth.png"), cv2.IMREAD_UNCHANGED)
    images.append((rgb_im, depth_im))

    rgb_im = cv2.imread(os.path.join(root, "img", "behind_color.png"), cv2.IMREAD_UNCHANGED)
    depth_im = cv2.imread(os.path.join(root, "img", "behind_depth.png"), cv2.IMREAD_UNCHANGED)
    images.append((rgb_im, depth_im))

    out = init_particle(images, bag_file = root + "record.bag")
