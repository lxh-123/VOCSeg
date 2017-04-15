"""Download data relevant to train the VOCSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import os
import subprocess

import zipfile


from six.moves import urllib
from shutil import copy2

import argparse

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

sys.path.insert(1, 'incl')

# Please set VOC_data_url to the download link for the VOC DATA.
#
# You can obtain by going to this website:
# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
#


vgg_url = 'ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy'


def get_pathes():
    """
    Get location of `data_dir` and `run_dir'.

    Defaut is ../DATA and ./RUNS.
    Alternativly they can be set by the environoment variabels
    'TV_DIR_DATA' and 'TV_DIR_RUNS'.
    """

    if 'TV_DIR_DATA' in os.environ:
        data_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        data_dir = "../DATA"

    if 'TV_DIR_RUNS' in os.environ:
        run_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        run_dir = "../RUNS"

    return data_dir, run_dir


def download(url, dest_directory):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    logging.info("Download URL: {}".format(url))
    logging.info("Download DIR: {}".format(dest_directory))

    def _progress(count, block_size, total_size):
                prog = float(count * block_size) / float(total_size) * 100.0
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, prog))
                sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath,
                                             reporthook=_progress)
    print()
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_url', default='', type=str)
    args = parser.parse_args()

    voc_data_url = args.voc_url

    data_dir, run_dir = get_pathes()

    vgg_weights = os.path.join(data_dir, 'weights', 'vgg16.npy')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download VGG DATA
    if not os.path.exists(vgg_weights):
        download_command = "wget {} -P {}".format(vgg_url, data_dir)
        logging.info("Downloading VGG weights.")
        download(vgg_url, data_dir)
    else:
        logging.warning("File: {} exists.".format(vgg_weights))
        logging.warning("Please delete to redownload VGG weights.")

    data_voc_zip = os.path.join(data_dir, 'VOCtrainval_11-May-2012.tar')

    # Download VOC DATA
    if not os.path.exists(data_voc_zip):
        if voc_data_url == '':
            logging.error("Data URL for VOC Data not provided.")
            url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
            logging.error("Please visit: {}".format(url))
            logging.error("and request VOC Download link.")
            logging.error("Rerun scipt using"
                          "'python download_data.py' --voc_url [url]")
            exit(1)
        if not voc_data_url[-19:] == 'VOCtrainval_11-May-2012.tar':
            logging.error("Wrong url.")
            url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
            logging.error("Please visit: {}".format(url))
            logging.error("and request VOC Download link.")
            logging.error("Rerun scipt using"
                          "'python download_data.py' --voc_url [url]")
            exit(1)
        else:
            logging.info("Downloading VOC2012 Data.")
            download(kitti_data_url, data_dir)

    # Extract and prepare VOC DATA
    logging.info("Extracting VOC data.")
    zipfile.ZipFile(data_voc_zip, 'r').extractall(data_dir)
    voc_dir = os.path.join(data_dir, 'VOCtrainval_11-May-2012/')

    logging.info("Preparing VOC data.")

    train_txt = "data/train.lst"
    val_txt = "data/val.lst"
    testing_txt = "data/test.lst"
    copy2(train_txt, voc_dir)
    copy2(val_txt, voc_dir)
    copy2(testing_txt, voc_dir)

    logging.info("All data have been downloaded successful.")


if __name__ == '__main__':
    main()
