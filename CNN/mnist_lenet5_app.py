#coding: utf-8
"""
-------------------------------------------------
   File Name：     mnist_lenet5_app
   Description :
   Author :       WANGFEI
   date：          2018/8/11
-------------------------------------------------
"""
import tensorflow as tf
import mnist_lenet5_forward
import mnist_lenet5_backwards
from PIL import Image
import numpy as np

def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x=tf.placeholder(tf.float32,shape=[1,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS])
        y=mnist_lenet5_forward.forward(x,False,None)
        preValue=tf.argmax(y,1)

        variable_averages=tf.train.ExponentialMovingAverage(mnist_lenet5_backwards.MOVING_AVERAGE_DECAY)
        variable_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backwards.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue=sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print('No file found')
                return -1

def pre_pics(testPic):
    img=Image.open(testPic)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))  # 变成灰度图
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
        nm_arr = im_arr.reshape([1,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS])
        nm_arr = nm_arr.astype(np.float32)
        img_ready = np.multiply(nm_arr, 1.0 / 255.0)
        return img_ready

def application():
    num=int(input('请输入图片的数量\n'))
    for i in range(num):
        testPic=input('请输入图片的位置\n')
        testPicArr=pre_pics(testPic)
        pre_value=restore_model(testPicArr)
        print('预测结果是：', pre_value)

def main():
    application()

if __name__ == '__main__':
    main()