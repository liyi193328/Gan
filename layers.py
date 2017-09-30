#encoding=utf-8
import tensorflow as tf

def linear(input, output_dim, scope=None, stddev=1.0, vis=True):
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
    if vis:
      tf.summary.histogram(w.name, w)
      tf.summary.histogram(b.name, b)

    return tf.matmul(input, w) + b