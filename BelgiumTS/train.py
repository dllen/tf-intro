# coding:utf8

import tensorflow as tf
from skimage import transform
from skimage.color import rgb2gray
import os
import loadData
import numpy as np
import random
import matplotlib.pyplot as plt

# 教程地址 https://www.jiqizhixin.com/articles/2017-07-30-3
# 数据下载地址
# http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip
# http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip
# 加载数据
ROOT_PATH = "../../"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = loadData.load_data(train_data_directory)
test_images, test_labels = loadData.load_data(test_data_directory)

images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
images28 = rgb2gray(images28)

test_images28 = [transform.resize(image, (28, 28)) for image in test_images]
test_images28 = rgb2gray(np.array(test_images28))

# 定义变量
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# 输入层
layers = tf.contrib.layers
images_flat = layers.flatten(x)

# 隐藏层
logits = layers.fully_connected(images_flat, 62, tf.nn.relu)

# 损失函数
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# 优化参数
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("image flat", images_flat)
print("logits", logits)
print("loss", loss)
print("accuracy", accuracy)

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10001):
        # print('epoch', i)
        _, loss_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 100 == 0:
            print("Loss : ", loss_val)
    # 测试数据集验证
    predicted_test = sess.run([correct_pred], feed_dict={x: test_images28})[0]
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted_test)])
    # Calculate the accuracy
    accuracy = match_count / len(test_labels)
    # Print the accuracy
    print("Accuracy: {:.3f}".format(accuracy))
    # Show sample images accuracy
    # 随机数据验证
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
    print('predicted ', predicted)
    plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:   {0}\nPrediction: {1}".format(truth, prediction), fontsize=12, color=color)
        plt.imshow(sample_images[i], cmap="gray")

plt.show()
