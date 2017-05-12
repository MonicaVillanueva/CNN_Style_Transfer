import tensorflow as tf
import os
from VGG import vgg16 as net
import utils
import numpy as np
import matplotlib.pyplot as plt


## Constants
VGG16_MEANS = [123.68, 116.779, 103.939]
PIC_SIZE = 224
CHANNELS = 3
ALPHA = 1
BETA = 100

INI_ETA = 1.5
DECAY = 0.95
STEPS = 500
N_EPOCHS = 10

## Paths
model_path = os.path.join('VGG', 'vgg16_weights.npz')
original_images_path = os.path.join('Dataset', 'original_input')
output_path = os.path.join(os.getcwd(), 'Test')
photo_path = os.path.join(original_images_path, 'neckarfront.jpg')
paint_path = os.path.join(original_images_path, 'van_gogh.jpg')


## Model
print("Load VGG model")
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, PIC_SIZE, PIC_SIZE, CHANNELS])
vgg = net.vgg16(imgs, model_path, sess)


## Images
photo = utils.read_image(photo_path, PIC_SIZE, PIC_SIZE, VGG16_MEANS)
paint = utils.read_image(paint_path, PIC_SIZE, PIC_SIZE, VGG16_MEANS)
# Create white noise image
mean = 0
std = 1
white_noise_img = utils.create_white_noise(mean, std, PIC_SIZE, PIC_SIZE, VGG16_MEANS)  # TODO: test with original mixed with noise?
white_noise = tf.Variable(tf.constant(np.array(white_noise_img, dtype=np.float32)), trainable=True, name='white_noise')
# plt.imshow(white_noise)


## Load images into the net
print("Get responses")
responses = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]
content_responses = [sess.run(res, feed_dict={vgg.imgs: [photo]}) for res in responses]
style_responses = [sess.run(res, feed_dict={vgg.imgs: [paint]}) for res in responses]
noise_responses = [sess.run(res, feed_dict={vgg.imgs: [white_noise_img]}) for res in responses]


print("Compute Gram Matrix")    # TODO: generalize to use any number of layers
style_gram_matrix = utils.compute_gram_matrix(sess, style_responses)
noise_gram_matrix = utils.compute_gram_matrix(sess, noise_responses)


## Train
print("Train")

# Compute content and style loss
cont_loss = utils.compute_content_loss(sess, content_responses, noise_responses)
style_loss = utils.compute_style_loss(sess, style_responses, style_gram_matrix, noise_gram_matrix)

# Compute total loss (alpha * content_loss + beta * style_loss)
loss = tf.Variable(ALPHA * cont_loss + BETA * style_loss, dtype=tf.float64)
# loss = ALPHA * cont_loss + BETA * style_loss

# Create tf optimizer
global_step = tf.Variable(0, trainable=False)
eta = tf.train.exponential_decay(learning_rate=INI_ETA, global_step=global_step, decay_steps=STEPS, decay_rate=DECAY, staircase=False)
train_step = tf.train.AdamOptimizer(eta).minimize(loss, global_step=global_step)
# optimizer = tf.train.AdamOptimizer(INI_ETA)   # No decay
# train_step = optimizer.minimize(loss)


## Optimize
print("Optimize")

sess.run(tf.global_variables_initializer())
for e in range(N_EPOCHS):

    # Feedback
    if e % 50 == 0:
        print('Epoch: ', e)
        print('Loss: ', sess.run(loss))
        img = sess.run(white_noise)
        utils.save_image(img, VGG16_MEANS, e, output_path)

    # Keep training
    print('Epoch: ', e)
    print('Loss: ', sess.run(loss))
    sess.run(train_step)


