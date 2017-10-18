# -*- coding: utf-8 -*-

import tensorflow as tf

#申明一个placeholder，理解为占位，shape可以填充时定，也可以事先定，run的时候填充数据
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]
