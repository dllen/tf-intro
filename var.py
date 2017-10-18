# -*- coding: utf-8 -*-

#tensorflow变量
import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])

#申明x变量，并初始化
x = tf.Variable([[1.0, 2.0]])
new_value = tf.add(x, matrix1)
#new_value赋值给x
update = tf.assign(x, new_value)

init_op = tf.initialize_all_variables()

sess = tf.Session()

with tf.Session() as sess:
  #需要先初始化
  sess.run(init_op)
  print(sess.run(x))
  for _ in range(3):
      #变量都需要通过sess.run执行
    sess.run(update)
    print(sess.run(x))
