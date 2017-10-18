# -*- coding: utf-8 -*-

#常量使用
import tensorflow as tf


# 创建两个常量，分别为1X2,2X1
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

# product等于两个矩阵乘积
product = tf.matmul(matrix1, matrix2)

# 启动session
sess = tf.Session()

# tf需要通过sess.run才能真实运行，并得到结果
with tf.Session() as sess:
  result = sess.run([product])
  print(result)
  print(sess.run(product))
