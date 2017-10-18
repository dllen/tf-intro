# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
# 生成测试数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
# 声明W、b变量，一个神经元，W随机初始化，-1.0到1.0取值
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
# loss函数定义为平方求平均
loss = tf.reduce_mean(tf.square(y - y_data))
# 采用梯度下降，学习率设为0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        # tf定义的变量都需要sess.run才可见
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]

