#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the VOCSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.analyze as ana
import tensorvision.utils as utils

from evaluation import voc_test

flags.DEFINE_string('RUN', 'VOCSeg_pretrained',
                    'Modifier for model parameters.')
flags.DEFINE_string('hypes', 'hypes/VOCSeg.json',
                    'File storing model parameters.')
flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))



def maybe_download_and_extract(runs_dir, filename):
    filepath = os.path.join(runs_dir, filename)

    if os.path.exists(filepath):
        # weights are downloaded. Nothing to do
        logging.info("Model parameters are ready")
        return

    else:
        logging.info("Model parameters are not found")
        return

def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                'VOCSeg')
    else:
        runs_dir = 'RUNS'

    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    train.maybe_download_and_extract(hypes)

    maybe_download_and_extract(runs_dir, "VOCSeg_2017_04_14_00.49")
    logging.info("Evaluating on Validation data.")
    logdir = os.path.join(runs_dir, "VOCSeg_2017_04_14_00.49")
    # logging.info("Output images will be saved to {}".format)
    # ana.do_analyze(logdir)

    logging.info("Creating output on test data.")
    voc_test.do_inference(logdir)

    logging.info("Analysis for pretrained model complete.")


if __name__ == '__main__':
    tf.app.run()
