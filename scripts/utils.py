import numpy as np
import torch
from torch.autograd import Variable
import os
import errno
from math import fabs, pi, log
import pickle
import random
import sys
from random import shuffle


def convert_to_variable(x, grad=True, gpu=True):
    if gpu:
        return Variable(torch.cuda.FloatTensor(x), requires_grad=grad)
    return Variable(torch.FloatTensor(x), requires_grad=grad)


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def normalize(x,
              n_file='normalization_factors.pkl',
              key='state'):
    fp = open(n_file, 'rb')
    normalization_factors = pickle.load(fp)
    fp.close()
    y = (x - normalization_factors[key][0]) / (normalization_factors[key][1] +
                                               1e-6)
    return y


def prepare_dir(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def distance(a, b, ignore_x=False, ignore_y=False, ignore_z=True):
    ans = 0
    if not ignore_x:
        ans += (a[0] - b[0]) ** 2
    if not ignore_y:
        ans += (a[1] - b[1]) ** 2
    if not ignore_z:
        ans += (a[2] - b[2]) ** 2
    ans = np.sqrt(ans)
    return ans


def preprocess_goal(v):
    norm = np.linalg.norm(v)
    v_norm = v / norm
    return v_norm


def find_norm(v):
    return np.linalg.norm(v)


def rotate_point(p, theta):
    x, y, z = p
    mod_x = x * np.cos(theta) - y * np.sin(theta)
    mod_y = x * np.sin(theta) + y * np.cos(theta)
    mod_z = z
    return np.array([mod_x, mod_y, mod_z])
