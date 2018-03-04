#encoding=utf-8
import tensorflow as tf

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
      initializer=tf.random_normal_initializer(stddev=stddev)
    )
    return tf.matmul(input, w) + b

def batch_normalize(x, is_training, decay=0.99, epsilon=0.001):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=True)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    return tf.cond(is_training, bn_train, bn_inference)


def batch_norm(x, is_training, epsilon=1e-3, momentum=0.99, name=None):
    """Code modification of http://stackoverflow.com/a/33950177"""
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                        center=True, scale=True, is_training=is_training, scope=name)

def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=True)
    return tf.add(tf.matmul(x, W), b)

def mlp(input, h_dim):
  init_const = tf.constant_initializer(0.0)
  init_norm = tf.random_normal_initializer()
  w0 = tf.get_variable('w0', [input.get_shape()[1], h_dim], initializer=init_norm)
  b0 = tf.get_variable('b0', [h_dim], initializer=init_const)
  w1 = tf.get_variable('w1', [h_dim, h_dim], initializer=init_norm)
  b1 = tf.get_variable('b1', [h_dim], initializer=init_const)
  h0 = tf.tanh(tf.matmul(input, w0) + b0)
  h1 = tf.tanh(tf.matmul(h0, w1) + b1)
  return h1


def generator(input, h_dim, feature_nums):
  transform, params = mlp(input, h_dim)
  init_const = tf.constant_initializer(0.0)
  init_norm = tf.random_normal_initializer()
  w = tf.get_variable('g_w', [h_dim, feature_nums], initializer=init_norm)
  b = tf.get_variable('g_b', [feature_nums], initializer=init_const)
  h = tf.matmul(transform, w) + b
  # s = tf.sigmoid(h)
  s = tf.tanh(h)
  return s, params + [w, b]


def minibatch(input, num_kernels=5, kernel_dim=3):
  x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
  activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
  diffs = tf.expand_dims(activation, 3) - \
          tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
  abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
  minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
  return tf.concat([input, minibatch_features], 1)


def discriminator1(input, h_dim):
  transform, params = mlp(input, h_dim)
  init_const = tf.constant_initializer(0.0)
  init_norm = tf.random_normal_initializer()
  w = tf.get_variable('d_w', [h_dim, 1], initializer=init_norm)
  b = tf.get_variable('d_b', [1], initializer=init_const)
  h_logits = tf.matmul(transform, w) + b
  h_prob = tf.sigmoid(h_logits)
  return h_prob, h_logits, params + [w, b]


# In[16]:

def optimizer(loss, var_list, num_decay_steps=1000):
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