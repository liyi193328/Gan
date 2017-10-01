#encoding=utf-8

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from Dataset import DataSet
import matplotlib.pyplot as plt

import layers
import shutil


activation_dict = {
  "tanh":tf.tanh,
  "sigmoid": tf.sigmoid
}

class AutoEncoder(object):

  def __init__(self, feature_num, hidden_size=None, learing_rate=0.01, activation="sigmoid", model_name="auto_encoder"):

    self.feature_num = feature_num
    if hidden_size is None:
      hidden_size = feature_num // 3
    self.hidden_size = hidden_size
    self.activation = activation_dict[activation]
    self.learning_rate = learing_rate
    self.model_name = model_name
    self._create_model()

  def _create_model(self):
    # tf Graph input (only pictures)
    self.X = tf.placeholder(tf.float32, [None, self.feature_num])
    self.mask = tf.placeholder(tf.float32, [None, self.feature_num])

    self.encoder_out = self.encoder(self.X)
    self.decoder_out = self.decoder(self.encoder_out)

    tf.summary.histogram("encoder_out", self.encoder_out)
    tf.summary.histogram("decoder_out", self.decoder_out)

    mask_decoder_out = self.decoder_out * self.mask

    total_valid_nums = tf.reduce_sum(self.mask)
    tf.summary.scalar("total_valid_nums", total_valid_nums)

    # Define loss and optimizer, minimize the squared error
    mask_mse = tf.reduce_sum( tf.pow(self.X - mask_decoder_out, 2) )
    self.loss = mask_mse / total_valid_nums

    # mask_mse = tf.reduce_sum(mask_mse, axis=1)
    # self.loss = tf.reduce_mean(mask_mse, axis=1)
    # self.loss = tf.reduce_mean(tf.pow(self.X - mask_decoder_out, 2))
    self.loss_sum = tf.summary.scalar("train_loss",self.loss)

    with tf.name_scope('optimizer'):
      # Gradient Descent
      optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
      # Op to calculate every variable gradient
      grads = tf.gradients(self.loss, tf.trainable_variables())
      grads = list(zip(grads, tf.trainable_variables()))
      # Op to update all variables according to their gradient
      self.apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

    # Create summaries to visualize weights
    for var in tf.trainable_variables():
      tf.summary.histogram(var.name, var)

    # Summarize all gradients
    for grad, var in grads:
      tf.summary.histogram(var.name + '/gradient', grad)

    # Merge all summaries into a single op
    self.merged_summary_op = tf.summary.merge_all()

    self.saver = tf.train.Saver(max_to_keep=1)

  def encoder(self, input):

    with tf.variable_scope("encoder"):

      out = layers.linear(input, self.hidden_size, scope="enc_first_layer")
      tf.summary.histogram("linear_out", out)
      out = self.activation(out)
      # out = layers.linear(out, self.hidden_size // 3, scope="enc_second_layer")
      # encoder_out = self.activation(out)

      #(None, fe) -> (None, fe // 3) -> tanh -> (None, fe // 9) -> tanh
    return out

  def decoder(self, input):

    with tf.variable_scope("decoder") as D:

      out = layers.linear(input, self.feature_num, scope="dec_first_layer")
      tf.summary.histogram("linear_out", out)
      out = self.activation(out)
      # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
      # decoder_out = self.activation(out)
      #(None, fe // 9) -> (None, fe // 3) -> tanh -> (None, fe) -> tanh
    return out


  def train_test_mnist(self, config):

    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Start Training
    # Start a new TF session
    with tf.Session() as sess:

      load_model_dir = os.path.join(config.checkpoint_dir, self.model_name)
      loaded = False
      if config.load_checkpoint and os.path.exists(load_model_dir):
        self.load(sess, config.checkpoint_dir)
        loaded = True

      elif os.path.exists(load_model_dir):
        shutil.rmtree(load_model_dir)

      # Run the initializer
      tf.global_variables_initializer().run()

      if not loaded:
        # Training
        for i in range(1, config.steps + 1):

          # Prepare Data
          # Get the next batch of MNIST data (only images are needed, not labels)
          batch_x, _ = mnist.train.next_batch(config.batch_size)
          mask = np.ones_like(batch_x)

          # Run optimization op (backprop) and cost op (to get loss value)
          _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.mask: mask})

          # Display logs per step
          if i % config.log_freq_steps == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

          if i % config.save_freq_steps == 0:
            save_dir = os.path.join(config.checkpoint_dir, self.model_name)
            self.save(sess, save_dir, i)

      # Testing
      # Encode and decode images from test set and visualize their reconstruction.
      n = 4
      canvas_orig = np.empty((28 * n, 28 * n))
      canvas_recon = np.empty((28 * n, 28 * n))
      for i in range(n):
          # MNIST test set
          batch_x, _ = mnist.test.next_batch(n)
          mask = np.ones_like(batch_x)
          # Encode and decode the digit image
          g = sess.run(self.decoder_out, feed_dict={self.X: batch_x, self.mask: mask})

          # Display original images
          for j in range(n):
              # Draw the original digits
              canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                  batch_x[j].reshape([28, 28])
          # Display reconstructed images
          for j in range(n):
              # Draw the reconstructed digits
              canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                  g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

  def train(self, config):

    dataset = DataSet(config.train_datapath, config.batch_size)
    steps = dataset.steps * config.epoch
    sample_dirs = os.path.join("samples", self.model_name)
    log_dirs = os.path.join("logs", self.model_name)

    for dir in [sample_dirs, log_dirs]:
      if os.path.exists(dir) == False:
        os.makedirs(dir)

    with tf.Session() as session:

      load_model_dir = os.path.join(config.checkpoint_dir, self.model_name)
      if config.load_checkpoint and os.path.exists(load_model_dir):
        self.load(session, config.checkpoint_dir)
      elif os.path.exists(load_model_dir):
        shutil.rmtree(load_model_dir)

      self.writer = tf.summary.FileWriter(log_dirs, session.graph)

      tf.global_variables_initializer().run()

      sample_batch = dataset.sample_batch()
      sample_mask = np.float32(sample_batch > 0.0)
      sample_path = os.path.join(sample_dirs, "{}.sample".format(self.model_name))
      pd.DataFrame(sample_batch).to_csv(sample_path, index=False)

      for step in range(steps):

        batch_data = dataset.next()
        mask = (batch_data > 0.0)
        mask = np.float32(mask)
        # print(np.shape(mask), mask.dtype)

        if step % config.save_freq_steps != 0:
          _, loss = session.run([self.apply_grads, self.loss],
                                           feed_dict={self.X: batch_data, self.mask: mask})
        else:
          _, summary_str, loss = session.run([self.apply_grads, self.merged_summary_op, self.loss],
                                             feed_dict={self.X: batch_data, self.mask: mask})

        if step % config.log_freq_steps == 0:
          print("step {}th, loss: {}".format(step, loss))

        if step % config.test_freq_steps == 0:
          predicts = session.run(self.decoder_out, feed_dict={self.X: sample_batch, self.mask: sample_mask})
          sample_path = os.path.join(sample_dirs, "{}.{}".format(self.model_name, step))
          pd.DataFrame(predicts, columns=dataset.columns).to_csv(sample_path, index=False)

        if step % config.save_freq_steps == 0:
          self.writer.add_summary(summary_str, step)
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
        mask = (batch_data > 0.0)
        mask = np.float32(mask)
        predicts = sess.run(self.decoder_out, feed_dict={self.X: batch_data, self.mask: mask})
        predict_data.append(predicts)
    predict_data = np.reshape(np.concatenate(predict_data, axis=0), (-1, dataset.feature_nums))
    df = pd.DataFrame(predict_data, columns=dataset.columns)
    if os.path.exists(config.outDir) == False:
      os.makedirs(config.outDir)
    outPath = os.path.join(config.outDir, "{}.infer.complete".format(self.model_name))
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