#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import scipy as scp


from seg_utils import seg_utils as seg
import time
import tensorvision
import tensorvision.utils as utils

def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0

    valid_gt = utils.load_segmentation_mask(hypes, gt_image)[:, :, :-1]

    return seg.evalExp(hypes, valid_gt, cnn_image)


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width), interp='cubic')
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width), interp='nearest')

    return image, gt_image


def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']
    eval_dict = {}
    num_classes = np.int32(hypes['arch']['num_classes'])
    color2class_dict = utils.get_class2color(hypes)

    for phase in ['train', 'val']:
        data_file = hypes['data']['{}_file'.format(phase)]
        data_file = os.path.join(data_dir, data_file)
        image_dir = os.path.dirname(data_file)

        # thresh = np.array(range(0, 256))/255.0
        total_intersection_class = np.zeros(num_classes)
        total_pixel_class = np.zeros(num_classes)
        total_unionsection_class = np.zeros(num_classes)

        image_list = []

        with open(data_file) as file:
            for i, datum in enumerate(file):
                    datum = datum.rstrip()
                    image_file, gt_file = datum.split(" ")
                    image_file = os.path.join(image_dir, image_file)
                    gt_file = os.path.join(image_dir, gt_file)

                    image = scp.misc.imread(image_file, mode='RGB')
                    gt_image = scp.misc.imread(gt_file, mode='RGB')

                    if hypes['jitter']['fix_shape']:
                        shape = image.shape
                        image_height = hypes['jitter']['image_height']
                        image_width = hypes['jitter']['image_width']
                        assert(image_height >= shape[0])
                        assert(image_width >= shape[1])

                        offset_x = (image_height - shape[0])//2
                        offset_y = (image_width - shape[1])//2
                        new_image = np.zeros([image_height, image_width, 3])
                        new_image[offset_x:offset_x+shape[0],
                                  offset_y:offset_y+shape[1]] = image
                        input_image = new_image
                    elif hypes['jitter']['reseize_image']:
                        image_height = hypes['jitter']['image_height']
                        image_width = hypes['jitter']['image_width']
                        gt_image_old = gt_image
                        image, gt_image = resize_label_image(image, gt_image,
                                                             image_height,
                                                             image_width)
                        input_image = image
                    else:
                        input_image = image

                    shape = input_image.shape

                    feed_dict = {image_pl: input_image}

                    output = sess.run([softmax], feed_dict=feed_dict)
                    output_im = output[0].reshape(shape[0], shape[1], num_classes)

                    if hypes['jitter']['fix_shape']:
                        gt_shape = gt_image.shape
                        output_im = output_im[offset_x:offset_x+gt_shape[0],
                                              offset_y:offset_y+gt_shape[1]]

                    if phase == 'val':
                        # Saving RB Plot
                        output_im_seg = np.argmax(output_im, axis=2)
                        ov_image = utils.overlay_segmentation(input_image, output_im_seg, color2class_dict)

                        name = os.path.basename(image_file)
                        image_list.append((name, ov_image))

                        # name2 = name.split('.')[0] + '_green.png'
                        # hard = output_im > 0.5
                        # green_image = utils.fast_overlay(image, hard)
                        # image_list.append((name2, green_image))

                    intersection_class, pixel_class, unionsection_class = eval_image(hypes, gt_image, output_im)

                    total_intersection_class += intersection_class
                    total_pixel_class += pixel_class
                    total_unionsection_class += unionsection_class

        eval_dict[phase] = seg.pxEval(total_intersection_class, total_pixel_class, total_unionsection_class)

        if phase == 'val':
            start_time = time.time()
            for i in xrange(10):
                sess.run([softmax], feed_dict=feed_dict)
            dt = (time.time() - start_time)/10

    eval_list = []

    for phase in ['train', 'val']:
        eval_list.append(('[{}] PixelAcc'.format(phase),
                          100*eval_dict[phase]['PixelAcc']))
        eval_list.append(('[{}] MeanPixelAcc'.format(phase),
                          100*eval_dict[phase]['MeanPixelAcc']))
        eval_list.append(('[{}] MeanIoU'.format(phase),
                          100*eval_dict[phase]['MeanIoU']))
        eval_list.append(('[{}] FwMeanIoU'.format(phase),
                          100 * eval_dict[phase]['FwMeanIoU']))

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list
