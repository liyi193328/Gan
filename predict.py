
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import codecs
import tensorflow as tf

import utils
import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import seaborn as sns
from IPython.display import HTML

from gan import Gan
from autoencoder import AutoEncoder

model_class_dict = {
  "Gan": Gan,
  "autoencoder": AutoEncoder
}

flags = tf.app.flags
flags.DEFINE_string("model_class", "autoencoder", "model class[autoencoder|gan]")

flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("feature_nums",13416, "The size of image to use")
flags.DEFINE_string("activation", "sigmoid", "auto encoder activation")

flags.DEFINE_string("infer_complete_datapath", "infer_complete_data.csv", "path of infer complete path")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

flags.DEFINE_string("model_name", "tanh-mak-0", "model name will make dir on checkpoint_dir")

FLAGS = flags.FLAGS

model_class = model_class_dict[FLAGS.model_class]
model = model_class(FLAGS.feature_nums, model_name=FLAGS.model_name)
model.train(FLAGS)
