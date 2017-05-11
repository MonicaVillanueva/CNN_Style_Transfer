import tensorflow as tf
import os
from VGG import vgg16 as net
import utils
import matplotlib.pyplot as plt


## Constants
PIC_SIZE = 224
CHANNELS = 3
ALPHA = 1
BETA = 100

## Paths
model_path = os.path.join('VGG', 'vgg16_weights.npz')
original_images_path = os.path.join('Dataset', 'original_input')
photo_path = os.path.join(original_images_path, 'neckarfront.jpg')
paint_path = os.path.join(original_images_path, 'van_gogh.jpg')


## Model
print("Load VGG model")
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, PIC_SIZE, PIC_SIZE, CHANNELS])
vgg = net.vgg16(imgs, model_path, sess)


## Images
photo = utils.read_image(photo_path, PIC_SIZE, PIC_SIZE)
paint = utils.read_image(paint_path, PIC_SIZE, PIC_SIZE)
# Create white noise image
mean = 0
std = 1
white_noise = utils.create_white_noise(mean, std, PIC_SIZE, PIC_SIZE)
# plt.imshow(white_noise)


## Load images into the net
print("Get responses")
responses = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]
content_responses = [sess.run(res, feed_dict={vgg.imgs: [photo]}) for res in responses]
style_responses = [sess.run(res, feed_dict={vgg.imgs: [paint]}) for res in responses]
noise_responses = [sess.run(res, feed_dict={vgg.imgs: [white_noise]}) for res in responses]


print("Compute Gram Matrix")# TODO: generalize to use any number of layers
style_gram_matrix = utils.compute_gram_matrix(sess, style_responses)
noise_gram_matrix = utils.compute_gram_matrix(sess, noise_responses)


## Train
print("Train")

# Compute content and style loss
cont_loss = utils.compute_content_loss(sess, content_responses, noise_responses)
style_loss = utils.compute_style_loss(sess, style_responses, style_gram_matrix, noise_gram_matrix)


# Compute total loss (alpha * content_loss + beta * style_loss)
loss = ALPHA * cont_loss + BETA * style_loss


