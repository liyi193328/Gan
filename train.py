
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
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("model_class", "autoencoder", "model class[autoencoder|gan]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("feature_nums",13416, "The size of image to use")
flags.DEFINE_string("activation", "sigmoid", "auto encoder activation")
flags.DEFINE_string("train_datapath", "./data/drop80-0-1.train", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("model_name", "auto_encoder0", "model name will make dir on checkpoint_dir")
flags.DEFINE_float("random_sample_mu",0.0,"random mu")
flags.DEFINE_float("random_sample_sigma", 1.0, "random sigma")

flags.DEFINE_integer("test_freq_steps", 100, "test freq steps")
flags.DEFINE_integer("save_freq_steps", 100, "save freq steps")
flags.DEFINE_integer("log_freq_steps", 10, "log freq steps")

flags.DEFINE_boolean("load_checkpoint", False, "if have checkpoint, whether load prev model first")
flags.DEFINE_integer("sample_steps", 500, "every sample_steps, will sample the generate mini batch data")
FLAGS = flags.FLAGS

model_class = model_class_dict[FLAGS.model_class]
model = model_class(FLAGS.feature_nums, model_name=FLAGS.model_name)
model.train(FLAGS)
