########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from VGG.imagenet_classes import class_names


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.data_dict = np.load(weights)
        print("Weights loaded")
        self.convlayers()

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name + '_W'], name="filter")

    def get_bias(self, name):
       return tf.constant(self.data_dict[name + '_b'], name="biases")

    def conv_layer(self, layer, previous_response):
        with tf.name_scope(layer) as scope:
            kernel = self.get_conv_filter(layer)
            conv = tf.nn.conv2d(previous_response, kernel, [1, 1, 1, 1], padding='SAME')

            biases = self.get_bias(layer)
            out = tf.nn.bias_add(conv, biases)
            result = tf.nn.relu(out, name=layer)
            setattr(self, layer, result)

    def pool_layer(self, layer, previous_response):
        result = tf.nn.max_pool(previous_response,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name=layer)
        setattr(self, layer, result)

    def convlayers(self):

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        self.conv_layer('conv1_1', images)
        self.conv_layer('conv1_2', self.conv1_1)
        self.pool_layer('pool1', self.conv1_2)

        self.conv_layer('conv2_1', self.pool1)
        self.conv_layer('conv2_2', self.conv2_1)
        self.pool_layer('pool2', self.conv2_2)

        self.conv_layer('conv3_1', self.pool2)
        self.conv_layer('conv3_2', self.conv3_1)
        self.conv_layer('conv3_3', self.conv3_2)
        self.pool_layer('pool3', self.conv3_3)

        self.conv_layer('conv4_1', self.pool3)
        self.conv_layer('conv4_2', self.conv4_1)
        self.conv_layer('conv4_3', self.conv4_2)
        self.pool_layer('pool4', self.conv4_3)

        self.conv_layer('conv5_1', self.pool4)
        self.conv_layer('conv5_2', self.conv5_1)
        self.conv_layer('conv5_3', self.conv5_2)
        self.pool_layer('pool5', self.conv5_3)


if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    responses_pool5 = sess.run(vgg.pool5, feed_dict={vgg.imgs: [img1]})
