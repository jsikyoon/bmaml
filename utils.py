import os 
import time 
import socket
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

# NOTE: this script is based on https://github.com/cbfinn/maml/blob/master/utils.py

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

def clip_if_not_none(grad, min_value, max_value):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, min_value, max_value)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_logdir(configs, fname_args=[]):
    this_run_str = time.strftime("%H%M%S_") + str(socket.gethostname())
    if is_git_dir():
        this_run_str += '_git' + git_hash_str() # random hash + git hash
    for str_arg in fname_args:
        if str_arg in configs.keys():
            this_run_str += '_' + str_arg.title().replace('_','') + '_' + str(configs[str_arg])
        else:
            raise ValueError('%s in fname_args does not exist in configs' % str_arg)
    this_run_str = this_run_str.replace('/','_')
    #log_dir = os.path.join(configs['log_root_dir'], configs['log_sub_dir'], this_run_str)
    return log_dir

def experiment_prefix_str(separator=',', hostname=False, git=True):
    this_run_str = time.strftime("%y%m%d_%H%M%S")
    if hostname:
        this_run_str += str(socket.gethostname())
    # NOTE: Unless you can attach your git folder when running borgy, this would fail!
    # Comment out the `is_git_dir` condition and the `str(git_hash_str())` to get this to work
    if git and is_git_dir():
        this_run_str += separator + str(git_hash_str()) # random hash + git hash 
    this_run_str = this_run_str.replace('-','')
    return this_run_str

def experiment_string2(configs, fname_args=[], separator=','):
    this_run_str = ''
    for (org_arg_str, short_arg_str) in fname_args:
        short_arg_str = org_arg_str.title().replace('_','') if short_arg_str is None else short_arg_str
        if org_arg_str in configs.keys():
            this_run_str += separator + short_arg_str + str(configs[org_arg_str]).title().replace('_','')
        else:
            raise ValueError('%s in fname_args doesn not exist in configs' % org_arg_str)
    this_run_str = this_run_str.replace('/', '_')
    return this_run_str

def experiment_string(configs, fname_args=[], separator=','):
    this_run_str = expr_prefix_str(configs)
    for str_arg in fname_args:
        if str_arg in configs.keys():
            this_run_str += separator + str_arg.title().replace('_','') + '=' + str(configs[str_arg])
        else:
            raise ValueError('%s in fname_args does not exist in configs' % str_arg)
    this_run_str = this_run_str.replace('/','_')
    return this_run_str

def is_git_dir():
    from subprocess import call, STDOUT
    if call(["git", "branch"], stderr=STDOUT, stdout=open(os.devnull, 'w')) != 0:
        return False
    else:
        return True

def git_hash_str(hash_len=7):
    import subprocess
    hash_str = subprocess.check_output(['git','rev-parse','HEAD'])
    return str(hash_str[:hash_len])

