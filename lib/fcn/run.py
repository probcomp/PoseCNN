# --------------------------------------------------------
# FCN
# Copyright (c) 2019 ProbComp at MIT
# Licensed under The MIT License [see LICENSE for details]
# Written by Austin Garrett
# --------------------------------------------------------
#
# A sensical handle for external call by Julia derendering library.

# TODO: Clean up these nonsense imports.
from fcn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from utils.voxelizer import Voxelizer, set_axes_equal
from utils.se3 import *
from utils.pose_error import *
from utils.bbox_transform import clip_boxes, bbox_transform_inv
from utils.nms import nms
import numpy as np
import cv2
import pickle
import os
import math
import tensorflow as tf
import time
from transforms3d.quaternions import quat2mat, mat2quat
import scipy.io
from scipy.optimize import minimize
from normals import gpu_normals
from .test import im_segment_single_frame, _extract_vertmap, vis_segmentations_vertmaps_detection


def run_network(sess, net, imdb, images, meta_data):
    """
    :param sess: TensorFlow session
    :param net: Pretrained neural network to run model over.
    :param imdb: TODO: Find out essential features of this object.
    :param images: [(rgb_image[0], depth_image[0]), ...]
    :param meta_data: Dictionary including camera intrinsics under 'intrinsic_matrix',
                      and scale factor under 'factor_depth' (default is 10,000).
    """

    n_images = len(images)
    segmentations = [[] for _ in range(n_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    # voxelizer
    voxelizer = Voxelizer(cfg.TEST.GRID_SIZE, imdb.num_classes)
    voxelizer.setup(-3, -3, -3, 3, 3, 4)

    # construct colors
    colors = np.zeros((3 * imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[i * 3 + 0] = imdb._class_colors[i][0]
        colors[i * 3 + 1] = imdb._class_colors[i][1]
        colors[i * 3 + 2] = imdb._class_colors[i][2]

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(n_images))
        # perm = xrange(n_images)
    else:
        perm = list(range(n_images))

    if (cfg.TEST.VERTEX_REG_2D and cfg.TEST.POSE_REFINE) or (cfg.TEST.VERTEX_REG_3D and cfg.TEST.POSE_REG):
        import libsynthesizer
        synthesizer = libsynthesizer.Synthesizer(cfg.CAD, cfg.POSE)
        synthesizer.setup(cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT)

    batched_detections = []

    for i in perm:

        raw_rgb, raw_depth = images[i]

        # read color image
        rgba = pad_im(raw_rgb, 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        im_depth = pad_im(raw_depth, 16)

        _t['im_segment'].tic()

        labels, probs, vertex_pred, rois, poses = im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, imdb._extents, imdb._points_all, imdb._symmetry, imdb.num_classes)

        detections = []

        for i in range(rois.shape[0]):
            cls_idx = int(rois[i, 1])
            if cls_idx > 0:
                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:7]
                cls = imdb._classes[cls_idx]
                detections.append((cls, RT))

        batched_detections.append(detections)

        labels = unpad_im(labels, 16)
        im_scale = cfg.TEST.SCALES_BASE[0]
        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        poses_new = []
        poses_icp = []
        if cfg.TEST.VERTEX_REG_2D:
            if cfg.TEST.POSE_REG:
                # pose refinement
                fx = meta_data['intrinsic_matrix'][0, 0] * im_scale
                fy = meta_data['intrinsic_matrix'][1, 1] * im_scale
                px = meta_data['intrinsic_matrix'][0, 2] * im_scale
                py = meta_data['intrinsic_matrix'][1, 2] * im_scale
                factor = meta_data['factor_depth']
                znear = 0.25
                zfar = 6.0
                poses_new = np.zeros((poses.shape[0], 7), dtype=np.float32)        
                poses_icp = np.zeros((poses.shape[0], 7), dtype=np.float32)     
                error_threshold = 0.01
                if cfg.TEST.POSE_REFINE:
                    labels_icp = labels.copy();
                    rois_icp = rois
                    if imdb.num_classes == 2:
                        I = np.where(labels_icp > 0)
                        labels_icp[I[0], I[1]] = imdb._cls_index
                        rois_icp = rois.copy()
                        rois_icp[:, 1] = imdb._cls_index
                    im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

                    parameters = np.zeros((7, ), dtype=np.float32)
                    parameters[0] = fx
                    parameters[1] = fy
                    parameters[2] = px
                    parameters[3] = py
                    parameters[4] = znear
                    parameters[5] = zfar
                    parameters[6] = factor

                    height = labels_icp.shape[0]
                    width = labels_icp.shape[1]
                    num_roi = rois_icp.shape[0]
                    channel_roi = rois_icp.shape[1]
                    synthesizer.icp_python(labels_icp, im_depth, parameters, height, width, num_roi, channel_roi, \
                                           rois_icp, poses, poses_new, poses_icp, error_threshold)

        _t['im_segment'].toc()

        _t['misc'].tic()
        labels_new = cv2.resize(labels, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_NEAREST)
        seg = {'labels': labels_new, 'rois': rois, 'poses': poses, 'poses_refined': poses_new, 'poses_icp': poses_icp}

        segmentations[i] = seg
        _t['misc'].toc()

        print(('im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i, n_images, _t['im_segment'].diff, _t['misc'].diff)))

        if cfg.TEST.VISUALIZE:
            img_dir = os.path.join("output", "vis")
            os.makedirs(img_dir, exist_ok = True)
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            vis_segmentations_vertmaps_detection(im, im_depth, im_label, imdb._class_colors, vertmap, 
                    labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'], imdb.num_classes, imdb._classes, imdb._points_all,
                    f_name = os.path.join(img_dir, "%i.png") % i)

    return batched_detections
