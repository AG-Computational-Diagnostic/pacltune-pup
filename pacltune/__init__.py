
from .utils import * ##

import os
print('#### Welcome to PaclTune :) ####')
conf = read_prj_conf()
print('################################')

import tensorflow as tf
physical_gpus = tf.config.list_physical_devices('GPU')
print_msg = 'Detected ' + str(len(physical_gpus)) + ' GPU(s).'
if len(physical_gpus) > 1:
    tf_strategy = tf.distribute.MirroredStrategy()
    print(print_msg + ' Using MirroredStrategy.')
else:
    tf_strategy = tf.distribute.get_strategy()
    print(print_msg + ' Using tf default strategy.')

# Set precision policy (https://www.tensorflow.org/guide/mixed_precision)
# Using mixed_float16 as a default (unlike tensorflow default of float32)
# Only good for devices with compute capability >= 7.0, REALLY SLOW FOR CPU!!
from tensorflow.keras.mixed_precision import experimental as mixed_precision
if 'tf_policy' in conf.keys():
    mixed_precision.set_policy(mixed_precision.Policy(conf['tf_policy']))
    print('Using \'tf_policy\' (=' + conf['tf_policy'] + ') from project.yml as policy.')
else:
    if len(physical_gpus) > 0:
        mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))
        print('Using mixed_float16 policy.')
    else:
        print('Using default policy since no GPU detected.')

from .model import * ##

defaults = Specification()

from .augment import * ##

AUTOTUNE = tf.data.experimental.AUTOTUNE

from .data import * ##
from .tune import * ##
from .predict import * ##
from .rmon import * ##
