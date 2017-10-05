
# coding: utf-8

import os
import sys
import plot

import pandas as pd
import numpy as np
import codecs
import tensorflow as tf

import utils
import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
from Dataset import DataSet
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
flags.DEFINE_integer("feature_nums",None, "The size of image to use")
flags.DEFINE_string("activation", "sigmoid", "auto encoder activation")
flags.DEFINE_string("train_datapath", "data/drop80-0-1.train", "Dataset directory.")
flags.DEFINE_string("infer_complete_datapath", "data/drop80-0-1.infer", "path of infer complete path")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("model_name", "auto_encoder0", "model name will make dir on checkpoint_dir")
flags.DEFINE_float("random_sample_mu",0.0,"random mu")
flags.DEFINE_float("random_sample_sigma", 1.0, "random sigma")

flags.DEFINE_boolean("plot_save", False, "plot fig after every test or prediction")
flags.DEFINE_integer("test_freq_steps", 500, "test freq steps")
flags.DEFINE_integer("save_freq_steps", 100, "save freq steps")
flags.DEFINE_integer("log_freq_steps", 10, "log freq steps")

flags.DEFINE_boolean("load_checkpoint", False, "if have checkpoint, whether load prev model first")
flags.DEFINE_integer("sample_steps", 500, "every sample_steps, will sample the generate mini batch data")

flags.DEFINE_string('outDir', 'prediction', "output dir")

FLAGS = flags.FLAGS

model_class = model_class_dict[FLAGS.model_class]
print("sampling dataset...")
sample_dataset = DataSet(FLAGS.train_datapath, nrows=20)
feature_nums = FLAGS.feature_nums or sample_dataset.feature_nums

model = model_class(feature_nums, model_name=FLAGS.model_name)
model.train(FLAGS)
