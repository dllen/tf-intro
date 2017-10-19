import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)
sess = tf.Session()

print(sess.run(y, feed_dict={a: 3.0, b: 3.0}))

num_puntos = 2000
conjunto_puntos = []
for i in range(num_puntos):
    if np.random.random() > 0.5:
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos], "y": [v[1] for v in conjunto_puntos]})
# sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
# plt.show()

vectors = tf.constant(conjunto_puntos)

k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)

axis = [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]) for c in range(k)]

print(axis)

means = tf.concat(0, axis=axis)

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

ess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(conjunto_puntos[i][0])
    data["y"].append(conjunto_puntos[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()

"""

Type in  Type in
                    Description
Python   TensorFlow
DT_FLOAT	tf.float32	Floating point of 32 bits
DT_INT16	tf.int16	Integer of 16 bits
DT_INT32	tf.int32	Integer of 32 bits
DT_INT64	tf.int64	Integer of 64 bits
DT_STRING	tf.string	String
DT_BOOL	tf.bool	Boolean
"""

"""
Shape	    Rank	Dimension Number
[]	           0	0-D
[D0]	       1	1-D
[D0, D1]	   2	2-D
[D0, D1, D2]   3	3-D
…	…	…
[D0, D1, … Dn]	n	n-D
"""

"""

Operation	Description
tf.shape	To find a shape of a tensor
tf.size	To find the size of a tensor
tf.rank	To find a rank of a tensor
tf.reshape	To change the shape of a tensor keeping the same elements contained
tf.squeeze	To delete in a tensor dimensions of size 1
tf.expand_dims	To insert a dimension to a tensor 
tf.slice	To remove a portions of a tensor
tf.split	To divide a tensor into several tensors along one dimension
tf.tile	To create a new tensor replicating a tensor multiple times
tf.concat	To concatenate tensors in one dimension
tf.reverse	To reverse a specific dimension of a tensor
tf.transpose	To transpose dimensions in a tensor
tf.gather	To collect portions according to an index
"""

"""

Operation	Description
tf.zeros_like	Creates a tensor with all elements initialized to 0
tf.ones_like	Creates a tensor with all elements initialized to 1
tf.fill	Creates a tensor with all elements initialized to a scalar value given as argument
tf.constant	Creates a tensor of constants with the elements listed as an arguments
"""

"""

Operation	Description
tf.random_normal	Random values with a normal distribution
tf.truncated_normal	Random values with a normal distribution but eliminating those values whose magnitude is more than 2 times the standard deviation
tf.random_uniform	Random values with a uniform distribution
tf.random_shuffle	Randomly mixed tensor elements in the first dimension
tf.set_random_seed	Sets the random seed
"""
