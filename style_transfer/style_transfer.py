import tensorflow as tf
import os
from VGG import vgg16 as net
from scipy.misc import imread, imresize

## Paths
model_path = os.path.join('VGG', 'vgg16_weights.npz')
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


## Load images into the net
print("Get content responses")
responses = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]
content_responses = [sess.run(res, feed_dict={vgg.imgs: [img1]}) for res in responses]

print("Get style responses")
style_responses = [sess.run(res, feed_dict={vgg.imgs: [img2]}) for res in responses]


print("Compute Gram Matrix")# TODO: generalize to use any number of layers
style_gram_matrix = []
for layer in range(len(style_responses)):
    # Vectorize response per filter
    num_filters = style_responses[layer].shape[3]
    vec_shape = [-1, num_filters]    # [-1] flattens into 1-D.
    vec_resp = tf.reshape(style_responses[layer], vec_shape)  # vectorize responses (to multiply)

    # Compute Gram Matrix as correlation one layer with itself
    corr = tf.matmul(tf.transpose(vec_resp), vec_resp)
    style_gram_matrix.append(sess.run(corr))


