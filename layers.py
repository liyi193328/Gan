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