#coding:utf-8
import tensorflow as tf
import numpy as np

BATCH_SIZE=8

normal=np.load('data/normalization.npy')
X=[i.reshape(136) for i in normal]
flag=np.load('data/np_flag.npy')
Y=[[i] for i in flag]
x=tf.placeholder(tf.float32,shape=(None,136))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([136,10],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([10,1],stddev=1,seed=1))
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEP = 3000
    for i in range(STEP):
        start = (i * BATCH_SIZE) % 415
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 200 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('%d轮后,loss为%g' % (i, total_loss))
    print('训练后的参数值为：\n')
    print('W1:\n', sess.run(w1))
    print('W2:\n', sess.run(w2))


