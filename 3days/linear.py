import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# hypothesis XW+b
hypothesis = x_train * W + b

# cost / loss function
# tf.square() 为取某个数的平方
# tf.reduce_mean() 为取均值
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 采用梯度下降更新权重
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 为了寻找能拟合数据的最好直线，我们需要最小化损失函数，即数据与直线之间的距离，因此我们可以采用梯度
train = optimizer.minimize(cost)

sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
