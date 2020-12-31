import os
import errno
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
#assert 与 if类似，不过是错误的时候执行后面的语句
EPS = 1e-7
def assert_eq(real, expected):
    assert real == expected, '%s(true) vs %s(expected)' %(real,expected)
def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)