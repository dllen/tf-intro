# coding:utf8

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
我们首先定义x和对应的正确标签y_。这里我们使用的是占位符，这样我们就能够动态扩展维度，并且只有在赋值时才真正申请空间。
None这一维是用来存一个batch的样本个数的。
'''
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 这里我们定义了两个变量来存储权重和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 输入经过简单的加权之后计算softmax的值作为每个种类的概率
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 接来是定义交叉熵代价函数，并申请了一个学习率为0.01的SGD的优化方法来学习参数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)
# 下面是计算准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    # 迭代1000轮
    for _ in range(1000):
        # batch大小为50
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    # 计算测试集准确率
    print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
