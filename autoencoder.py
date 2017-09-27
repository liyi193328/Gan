#encoding=utf-8

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from Dataset import DataSet

import layers
import shutil

activation_dict = {
  "tanh":tf.tanh,
  "sigmoid": tf.sigmoid
}

class AutoEncoder(object):

  def __init__(self, feature_num, hidden_size, learing_rate=0.01, activation="sigmoid", model_name="auto_encoder"):

    self.feature_num = feature_num
    self.hidden_size = hidden_size
    self.activation = activation_dict[activation]
    self.learning_rate = learing_rate
    self.model_name = model_name
    self._create_model()

  def _create_model(self):
    # tf Graph input (only pictures)
    self.X = tf.placeholder("float", [None, self.feature_num])
    self.encoder_out = self.encoder(self.X)
    self.decoder_out = self.decoder(self.encoder_out)

    # Define loss and optimizer, minimize the squared error
    self.loss = tf.reduce_mean(tf.pow(self.X - self.decoder_out, 2))
    self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    self.saver = tf.train.Saver(max_to_keep=1)

  def encoder(self, input):

    with tf.VariableScope("encoder") as E:

      out = layers.linear(input, self.hidden_size)
      encoder_out = self.activation(out)

    return encoder_out

  def decoder(self, input):

    with tf.VariableScope("decoder") as D:

      out = layers.linear(input, self.feature_num)
      decoder_out = self.activation(out)

    return decoder_out


  def train(self, config):

    dataset = DataSet(config.train_datapath, config.batch_size)
    steps = dataset.steps * config.epoch
    sample_dirs = os.path.join("samples", self.model_name)


    with tf.Session() as session:

      if config.load_checkpoint and os.path.exists(config.checkpoint_dir):
        self.load(session, config.checkpoint_dir)
      elif os.path.exists(config.checkpoint_dir):
        shutil.rmtree(config.checkpoint_dir)

      tf.global_variables_initializer().run()

      for step in range(steps):

        batch_data = dataset.next()
        _, loss = session.run([self.optimizer, self.loss], feed_dict={self.X: batch_data})

        if step % config.log_every == 0:
          print("step {}th, loss: {}".format(loss))

        if step % config.test_freq == 0:
          predicts = session.run(self.decoder_out, feed_dict={self.X: batch_data})
          sample_path = os.path.join(sample_dirs, "{}.{}".format(self.model_name, step))
          pd.DataFrame(predicts).to_csv(sample_path, index=False, header=None)

        if step % config.save_freq_steps == 0:
          save_dir = os.path.join(config.checkpoint_dir, self.model_name)
          self.save(session, save_dir, step)

  def predict(self, config):
    dataset = DataSet(config.infer_complete_datapath, batch_size=config.batch_size, onepass=True)
    predict_data = []
    with tf.Session() as sess:
      load_model_dir = os.path.join(config.checkpoint_dir, self.model_name)
      isLoaded = self.load(sess, load_model_dir)
      assert (isLoaded)
      try:
        tf.global_variables_initializer().run()
      except:
        tf.initialize_all_variables().run()
      while (1):
        batch_data = dataset.next()
        if batch_data is None:
          break
        predicts = sess.run(self.decoder_out, feed_dict={self.X: batch_data})
        predict_data.append(predicts)
    predict_data = np.reshape(np.concatenate(predict_data, axis=0), (-1, dataset.feature_nums))
    df = pd.DataFrame(predict_data)
    if os.path.exists(config.outDir) == False:
      os.makedirs(config.outDir)
    outPath = os.path.join(config.outDir, "infer.complete")
    df.to_csv(outPath, index=None)
    print("save complete data from {} to {}".format(config.infer_complete_datapath, outPath))

  def save(self, sess, save_dir, step):
      if not os.path.exists(save_dir):
          os.makedirs(save_dir)

      self.saver.save(sess,
                      os.path.join(save_dir, self.model_name),
                      global_step=step)

  def load(self, sess, checkpoint_dir):
      print(" [*] Reading checkpoints...")

      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
          self.saver.restore(sess, ckpt.model_checkpoint_path)
          return True
      else:
          return False