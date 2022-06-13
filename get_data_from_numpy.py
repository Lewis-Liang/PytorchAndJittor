import numpy as np
import random
import math


def np_fix_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

def get_data(b,c,w,h):
    return np.random.normal(loc=1.0,scale=0.5,size=(b,c,h,w)).astype(np.float32)

def get_numpy_weight(initialization, out_dim, in_dim):
    if initialization == 'zeros':
        return np.zeros([out_dim, in_dim]), np.zeros([out_dim])
    elif initialization == 'ones':
        return np.ones([out_dim, in_dim]), np.ones([out_dim])
    elif initialization == 'normal':
        return np.random.normal(loc=1., scale=0.02, size=[out_dim, in_dim]), \
                np.random.normal(loc=1., scale=0.02, size=[out_dim])
    elif initialization == 'xavier_Glorot_normal':
        return np.random.normal(loc=0., scale=1., size=[out_dim, in_dim]) / np.sqrt(in_dim), \
                np.random.normal(loc=0., scale=1., size=[out_dim]) / np.sqrt(in_dim)
    elif initialization == 'xavier_normal':
        matsize = out_dim * in_dim
        fan = (out_dim * matsize) + (in_dim * matsize)
        std = 0.02 * math.sqrt(2.0/fan)
        return np.random.normal(loc=0., scale=std, size=[out_dim, in_dim]), \
                np.random.normal(loc=0., scale=std, size=[out_dim])
    elif initialization == 'uniform':
        a = np.sqrt(1. / in_dim)
        return np.random.uniform(low=-a, high=a, size=[out_dim, in_dim]), \
                np.random.uniform(low=-a, high=a, size=[out_dim])
    elif initialization == 'xavier_uniform':
        a = np.sqrt(6. / (in_dim + out_dim))
        return np.random.uniform(low=-a, high=a, size=[out_dim, in_dim]), \
                np.random.uniform(low=-a, high=a, size=[out_dim])