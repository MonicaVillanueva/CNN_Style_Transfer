import tensorflow as tf
import os
from VGG import vgg16 as net
from scipy.misc import imread, imresize

## Paths
model_path = os.path.join('model', 'vgg16_weights.npz')
original_images_path = os.path.join('Dataset', 'original_input')
photo_path = os.path.join(original_images_path, 'neckarfront.jpg')
paint_path = os.path.join(original_images_path, 'van_gogh.jpg')


## Model
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = net.vgg16(imgs, model_path, sess)


## Input images
img1 = imread(photo_path, mode='RGB')
img1 = imresize(img1, (224, 224))

img2 = imread(paint_path, mode='RGB')
img2 = imresize(img1, (224, 224))


## Load images
