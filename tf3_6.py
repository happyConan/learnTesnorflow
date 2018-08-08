#coding:utf-8
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

BATCH_SIZE=8
seed=23455

rng=np.random.RandomState(seed)
X=rng.rand(32,2)
Y=[[int(x0+x1<1)] for [x0,x1] in X]
print('X:\n',X)
print('Y:\n',Y)

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#定义损失函数
loss=tf.reduce_mean(tf.square(y-y_))  #均方误差的方法
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)#0.001是学习率
#还有其他自带的优化器
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print('W1:\n',sess.run(w1))
    print('W2:\n', sess.run(w2))

    STEP=3000
    for i in range(3000):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%500==0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print('%d轮后，loss为： %g'%(i,total_loss))
    print('训练后的参数值为：\n')
    print('W1:\n', sess.run(w1))
    print('W2:\n', sess.run(w2))