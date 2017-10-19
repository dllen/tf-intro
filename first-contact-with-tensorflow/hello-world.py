import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)

sess = tf.Session()

print(sess.run(y, feed_dict={a: 3, b: 3}))

"""

Operation	Description
tf.add	sum
tf.sub	substraction
tf.mul	multiplication
tf.div	division
tf.mod	module
tf.abs	return the absolute value
tf.neg	return negative value
tf.sign	return the sign
tf.inv	returns the inverse
tf.square	calculates the square
tf.round	returns the nearest integer
tf.sqrt	calculates the square root
tf.pow	calculates the power
tf.exp	calculates the exponential
tf.log	calculates the logarithm
tf.maximum	returns the maximum
tf.minimum	returns the minimum
tf.cos	calculates the cosine
tf.sin	calculates the sine
"""

"""

Operation	Description
tf.diag	returns a diagonal tensor with a given diagonal values
tf.transpose	returns the transposes of the argument
tf.matmul	returns a tensor product of multiplying two tensors listed as arguments
tf.matrix_determinant	returns the determinant of the square matrix specified as an argument
tf.matrix_inverse	returns the inverse of the square matrix specified as an argument
"""

"""

Operations groups	Operations
Maths	Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
Array	Concat, Slice, Split, Constant, Rank, Shape, Shuffle
Matrix	MatMul, MatrixInverse, MatrixDeterminant
Neuronal Network	SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool
Checkpointing	Save, Restore
Queues and syncronizations	Enqueue, Dequeue, MutexAcquire, MutexRelease
Flow control	Merge, Switch, Enter, Leave, NextIteration
"""
