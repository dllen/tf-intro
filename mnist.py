# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# 创建模型，x是放置输入样本集，每个训练样本784个点
x = tf.placeholder(tf.float32, [None, 784])
# 定义10个神经元，
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 求softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_样本结果集
y_ = tf.placeholder(tf.float32, [None, 10])
# loss函数定义
# 交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
# tf.initialize_all_variables().run()
tf.global_variables_initializer().run()
for i in range(1000):
    # 每次取100个，随机取，迭代1000次
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 比较最大的一个是否是同一个
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 求评价准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
