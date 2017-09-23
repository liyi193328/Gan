
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import codecs
import tensorflow as tf

import shutil
import utils
import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import seaborn as sns
from IPython.display import HTML

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

def mlp(input, h_dim):
    init_const = tf.constant_initializer(0.0)
    init_norm = tf.random_normal_initializer()
    w0 = tf.get_variable('w0', [input.get_shape()[1], h_dim], initializer=init_norm)
    b0 = tf.get_variable('b0', [h_dim], initializer=init_const)
    w1 = tf.get_variable('w1', [h_dim, h_dim], initializer=init_norm)
    b1 = tf.get_variable('b1', [h_dim], initializer=init_const)
    h0 = tf.tanh(tf.matmul(input, w0) + b0)
    h1 = tf.tanh(tf.matmul(h0, w1) + b1)
    return h1, [w0, b0, w1, b1]

def generator(input, h_dim, feature_nums):
    transform, params = mlp(input, h_dim)
    init_const = tf.constant_initializer(0.0)
    init_norm = tf.random_normal_initializer()
    w = tf.get_variable('g_w', [h_dim, feature_nums], initializer=init_norm)
    b = tf.get_variable('g_b', [feature_nums], initializer=init_const)
    h = tf.matmul(transform, w) + b
    s = tf.sigmoid(h)
    return s, params + [w, b]

def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b

def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

'''
def discriminator(input, h_dim, minibatch_layer=False):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3
'''

def discriminator(input, h_dim):
    transform, params = mlp(input, h_dim)
    init_const = tf.constant_initializer(0.0)
    init_norm = tf.random_normal_initializer()
    w = tf.get_variable('d_w', [h_dim, 1], initializer=init_norm)
    b = tf.get_variable('d_b', [1], initializer=init_const)
    h = tf.sigmoid(tf.matmul(transform, w) + b)
    return h, params + [w, b]


# In[16]:

def optimizer(loss, var_list, num_decay_steps = 1000):
    initial_learning_rate = 0.01
    decay = 0.95
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer

class DataSet:
    
    def __init__(self, path, batch_size=128, shuffle=True, onepass=False):
        print("make dataset from {}...".format(path))
        data = pd.read_csv(path, sep=",").values
        self.path = path
        self.data = data
        self.samples, self.feature_nums = data.shape
        self.cnt = 0
        self.batch_counter = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.onepass = onepass
        print("batch_size is {}, have {} samples, {} features, step nums is {}".format(batch_size, self.samples, self.feature_nums, self.steps))
        print("make dataset end")

    def next(self):

        batch_size = self.batch_size

        if self.cnt >= self.samples and self.onepass is True: #for infer mode
            return None

        if self.cnt + batch_size >= self.samples:
            if self.onepass: # if last pass piece, make batch_data
                batch_data = self.data[self.cnt:]
                self.cnt = self.samples
                print("the last batch, shape is {}...".format(batch_data.shape))
                return batch_data

            self.cnt = 0
            self.shuffle_data()

        be, en = self.cnt, min(self.samples, self.cnt + batch_size)
#         yield data[be, en]
        batch_data = self.data[be : en]
        self.cnt = (self.cnt + batch_size) % self.samples
        self.batch_counter += 1
        print("getting {}th batch end".format(self.batch_counter))
        return batch_data

    def shuffle_data(self):
        np.random.shuffle(self.data)

    @property
    def steps(self):
        return self.samples // self.batch_size

anim_frames = []

def plot_distributions(GAN, session, loss_d, loss_g):
    num_points = 100000
    num_bins = 100
    xs = np.linspace(-GAN.gen.range, GAN.gen.range, num_points)
    bins = np.linspace(-GAN.gen.range, GAN.gen.range, num_bins)

    # p(data)
    d_sample = GAN.data.sample(num_points)

    # decision boundary
    ds = np.zeros((num_points, 1))  # decision surface
    for i in range(num_points // GAN.batch_size):
        ds[GAN.batch_size * i:GAN.batch_size * (i + 1)] = session.run(GAN.D1, {
            GAN.x: np.reshape(xs[GAN.batch_size * i:GAN.batch_size * (i + 1)], (GAN.batch_size, 1))
        })

    # p(generator)
    zs = np.linspace(-GAN.gen.range, GAN.gen.range, num_points)
    gs = np.zeros((num_points, 1))  # generator function
    for i in range(num_points // GAN.batch_size):
        gs[GAN.batch_size * i:GAN.batch_size * (i + 1)] = session.run(GAN.G, {
            GAN.z: np.reshape(
                zs[GAN.batch_size * i:GAN.batch_size * (i + 1)],
                (GAN.batch_size, 1)
            )
        })
           
    anim_frames.append((d_sample, ds, gs, loss_d, loss_g))



class DCGAN(object):

    def __init__(self, feature_nums, mlp_hidden_size=2000, lam=0.1):

        self.feature_nums = feature_nums
        self.log_every = 10
        self.mlp_hidden_size = mlp_hidden_size
        self.lam = lam
        self.model_name = "DCGAN.model"
        self._create_model()

    def _create_model(self):

        self.is_training = tf.placeholder(tf.bool, name="is_trainging")
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(None, self.feature_nums))
            self.G, theta_g = generator(self.z, self.mlp_hidden_size, self.feature_nums)
            self.z_sum = tf.summary.histogram("z", self.z)

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=(None, self.feature_nums))
            self.D1, self.theta_d1 = discriminator(self.x, self.mlp_hidden_size)
            scope.reuse_variables()
            self.D2, self.theta_d2 = discriminator(self.G, self.mlp_hidden_size)

        self.d_sum = tf.summary.histogram("d1", self.D1)
        self.d__sum = tf.summary.histogram("d_", self.D2)
        self.G_sum = tf.summary.histogram("G", self.G)

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1, labels=tf.ones_like(self.D1)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2, labels=tf.zeros_like(self.D2)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.loss_d = self.d_loss_real + self.d_loss_fake
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2, labels=tf.zeros_like(self.D2)))

        self.g_loss_sum = tf.summary.scalar("g_loss", self.loss_g)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.loss_d)

        self.opt_d = optimizer(self.loss_d, self.theta_d2)
        self.opt_g = optimizer(self.loss_g, theta_g)

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, [None, self.feature_nums], name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.x))), 1)
        self.perceptual_loss = self.loss_g
        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)
    
    def train(self, config):

        dataset = DataSet(config.train_datapath, config.batch_size)

        steps = dataset.steps * config.epoch

        samples = np.random.normal(config.random_sample_mu, config.random_sample_sigma,
                                        (config.batch_size, self.feature_nums))


        with tf.Session() as session:

            if config.load_checkpoint and os.path.exists(config.checkpoint_dir):
                self.load(session, config.checkpoint_dir)
            elif os.path.exists(config.checkpoint_dir):
                shutil.rmtree(config.checkpoint_dir)

            tf.global_variables_initializer().run()

            self.g_sum = tf.summary.merge(
                [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
            self.d_sum = tf.summary.merge(
                [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

            self.writer = tf.summary.FileWriter("./logs", session.graph)

            for step in range(steps):
                
                batch_data = dataset.next()

                sz = len(batch_data)

                random_data = np.random.normal(0, 1,(sz, self.feature_nums))

                loss_d, _ , d_summary_str = session.run([self.loss_d, self.opt_d, self.d_sum], {
                    self.x: batch_data,
                    self.z: random_data
                })

                self.writer.add_summary(d_summary_str, steps)

                # update generator
                loss_g, _ , g_summary_str = session.run([self.loss_g, self.opt_g, self.g_sum], {
                    self.z: random_data
                })
                self.writer.add_summary(g_summary_str, steps)

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))

                if step % config.save_freq_steps == 0:
                    self.save(session, config.checkpoint_dir, step)


    def complete(self, config):

        dataset = DataSet(config.infer_complete_datapath, batch_size=config.batch_size, onepass=True)

        missing_val = config.missing_val

        complete_datas = []
        feature_nums = dataset.feature_nums

        with tf.Session() as sess:

            isLoaded = self.load(sess, config.checkpoint_dir)
            assert (isLoaded)

            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            while(1):
                batch_data = dataset.next()
                if batch_data is None:
                    break
                data_shape = np.shape(batch_data)
                sample_size, feature_nums = data_shape

                batch_mask = utils.MaskData(batch_data, missing_val)
                mask_data = np.multiply(batch_data, batch_mask)
                zhats = np.random.uniform(0, 1, size=data_shape)
                completed = batch_data

                m = 0
                v = 0
                G_data = None

                for i in range(config.nIter):
                    fd = {
                        self.z: zhats,
                        self.mask: batch_mask,
                        self.x: batch_data,
                        self.is_training: False
                    }
                    run = [self.complete_loss, self.grad_complete_loss, self.G]
                    loss, g, G_data = sess.run(run, feed_dict=fd)

                    if config.approach == 'adam':
                        # Optimize single completion with Adam
                        m_prev = np.copy(m)
                        v_prev = np.copy(v)
                        m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                        v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                        m_hat = m / (1 - config.beta1 ** (i + 1))
                        v_hat = v / (1 - config.beta2 ** (i + 1))
                        zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                        zhats = np.clip(zhats, -1, 1)

                    elif config.approach == 'hmc':
                        # Sample example completions with HMC (not in paper)
                        zhats_old = np.copy(zhats)
                        loss_old = np.copy(loss)
                        v = np.random.randn(sample_size, feature_nums)
                        v_old = np.copy(v)

                        for steps in range(config.hmcL):
                            v -= config.hmcEps/2 * config.hmcBeta * g[0]
                            zhats += config.hmcEps * v
                            np.copyto(zhats, np.clip(zhats, -1, 1))
                            loss, g, _, _ = sess.run(run, feed_dict=fd)
                            v -= config.hmcEps/2 * config.hmcBeta * g[0]

                        for i in range(sample_size):
                            logprob_old = config.hmcBeta * loss_old[i] + np.sum(v_old[i]**2)/2
                            logprob = config.hmcBeta * loss[i] + np.sum(v[i]**2)/2
                            accept = np.exp(logprob_old - logprob)
                            if accept < 1 and np.random.uniform() > accept:
                                np.copyto(zhats[i], zhats_old[i])

                        config.hmcBeta *= config.hmcAnneal

                inv_masked_hat_data = np.multiply(G_data, 1.0 - batch_mask)
                completed = mask_data + inv_masked_hat_data
                complete_datas.append(completed)

        complete_datas = np.reshape(np.concatenate(complete_datas,axis=0), (-1, feature_nums))
        df = pd.DataFrame(complete_datas)
        if os.path.exists(config.outDir) == False:
            os.makedirs(config.outDir)
        outPath = os.path.join(config.outDir, "infer.complete")
        df.to_csv(outPath, index=None)
        print("save complete data from {} to {}".format(config.infer_complete_datapath, outPath))

    def save(self, sess, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False