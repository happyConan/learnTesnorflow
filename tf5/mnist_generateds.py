import tensorflow as tf
import os
from PIL import Image

img_train_path='./mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path='./mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train='./data/mnist_train.tfRecords'
img_test_path='images/'
label_test_path='test_jpg.txt'
tfRecord_test='data/mnist_test.tfRecords'
data_path='data/'
resize_height=28
resize_width=28

def read_tfRecord(tfRecord_path):
    filename_queue=tf.train.string_input_producer([tfRecord_path])
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features=tf.parse_single_example(serialized_example,features={'label':tf.FixedLenFeature([10],tf.int64),'img_raw':tf.FixedLenFeature([],tf.string)})
    img=tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img, shape=[784])
    img=tf.cast(img,tf.float32)*(1./255)
    label=tf.cast(features['label'],tf.float32)
    print(img,label)
    return img,label

def get_tfrecord(num,isTrain=True):
    if isTrain:
        tfRecord_path=tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    print(tfRecord_path)
    img,label=read_tfRecord(tfRecord_path)
    print('++++++++++++++++++++++++++++++++++1')
    img_batch,label_batch=tf.train.shuffle_batch([img,label],batch_size=num,num_threads=2,capacity=35,min_after_dequeue=5)
    return img_batch,label_batch


def write_tfRecord(tfRecordName,image_path,label_path):
    writer=tf.python_io.TFRecordWriter(tfRecordName)
    num_pic=0
    f=open(label_path,'r')
    contents=f.readlines()
    f.close()
    for content in contents:
        value=content.split()
        img_path=image_path+value[0]
        img=Image.open(img_path)
        img_raw=img.tobytes()
        labels=[0]*10
        labels[int(value[1])]=1
        example=tf.train.Example(features=tf.train.Features(feature={'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}))
        writer.write(example.SerializeToString())
        num_pic+=1
        print('The number of picture:',num_pic)
    writer.close()
    print('write tfRecord successfully')

def generate_tfRecord():
    isExists=os.path.exists(data_path)
    if not isExists:
        os.mkdir(data_path)
        print('the directory is created successfully')
    else:
        print('directory already exists')
    #write_tfRecord(tfRecord_train,img_train_path,label_train_path)
    write_tfRecord(tfRecord_test, img_test_path, label_test_path)

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()