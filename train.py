
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

from model import DCGAN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("feature_nums",11246, "The size of image to use")
flags.DEFINE_string("train_datapath", "../data_preprocessed/drop80.train", "Dataset directory.")
flags.DEFINE_string("infer_complete_datapath", "infer_complete_data.csv", "path of infer complete path")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_float("random_sample_mu",0.0,"random mu")
flags.DEFINE_float("random_sample_sigma", 1.0, "random sigma")
flags.DEFINE_integer("save_freq_steps", 100, "save freq steps")
flags.DEFINE_boolean("load_checkpoint", False, "if have checkpoint, whether load prev model first")
FLAGS = flags.FLAGS

model = DCGAN(FLAGS.feature_nums)
model.train(FLAGS)
