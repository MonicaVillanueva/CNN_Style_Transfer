import tensorflow as tf
import os
import numpy as np
from datetime import datetime as dt
from VGG import vgg16 as net
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt


def save_image(im, iteration, out_dir):
    img = im.copy()
    img = np.clip(img,0,255).astype(np.uint8) # img[0, ...] ???
    nowtime = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    imsave("{}\pneural_art_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)

# Parameters
alpha = tf.constant(1.0,name='alpha')
beta = tf.constant(100.0,name='beta')
num_iters = 1000


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
img2 = imresize(img2, (224, 224))
# plt.imshow(img2)
# plt.show()

## Load images into the net
print("Get content responses")
content_constant = tf.constant(img1, dtype=tf.float32, name="content_constant")
vgg_content = net.vgg16(content_constant, model_path, sess)
content_responses = [vgg_content.conv1_1, vgg_content.conv2_1, vgg_content.conv3_1, vgg_content.conv4_1, vgg_content.conv5_1]

print("Get style responses")
style_constant = tf.constant(img2, dtype=tf.float32)
vgg_style = net.vgg16(style_constant, model_path, sess)
style_responses = [vgg_style.conv1_1, vgg_style.conv2_1, vgg_style.conv3_1, vgg_style.conv4_1, vgg_style.conv5_1]

print("Compute Gram Matrix")# TODO: generalize to use any number of layers
style_gram_matrix = []
for layer in range(len(style_responses)):
    # Vectorize response per filter
    num_filters = style_responses[layer].get_shape().as_list()[3]
    vec_shape = [-1, num_filters]    # [-1] flattens into 1-D.
    vec_resp = tf.reshape(style_responses[layer], vec_shape)  # vectorize responses (to multiply)
    # Compute Gram Matrix as correlation one layer with itself
    corr = tf.matmul(tf.transpose(vec_resp), vec_resp, name='corr')
    style_gram_matrix.append(corr)

# Build graph
L_content = tf.Variable(0.0, name="L_content_var")
L_style = tf.Variable(0.0, name="L_style")

# Initialize noisy image
# gen_img = tf.Variable(tf.truncated_normal(img1.shape, stddev=20, dtype=tf.float32), dtype=tf.float32, trainable=True, name='gen_img')
gen_img = tf.Variable(img1, dtype=tf.float32, name="gen_img")
global_step = tf.Variable(0, trainable=False, name='global_step')



vgg_aux = net.vgg16(gen_img, model_path, sess)
gen_img_responses = [vgg_aux.conv1_1, vgg_aux.conv2_1, vgg_aux.conv3_1, vgg_aux.conv4_1, vgg_aux.conv5_1]


print('After image')
for layer in range(len(gen_img_responses)):
    # Content loss
    L_content += tf.nn.l2_loss(content_responses[layer] - gen_img_responses[layer], name='L_content')
    # Style
    num_filters = style_responses[layer].get_shape().as_list()[3]
    vec_shape = [-1, num_filters]  # [-1] flattens into 1-D.
    vec_resp = tf.reshape(gen_img_responses[layer], vec_shape, name='vec_resp')  # vectorize responses (to multiply)

    # Compute Gram Matrix as correlation one layer with itself
    corr = tf.matmul(tf.transpose(vec_resp), vec_resp, name='corr')
    N = np.prod(content_responses[layer].get_shape().as_list()).astype(np.float32)
    # N is the product of the 3 dimensions of the responses, because we want N = number_filters * size(feature_map)
    norm_factor = tf.constant((2 * (N**2)) / len(content_responses), name='norm_factor', dtype='float32')
    L_style += tf.nn.l2_loss(corr - style_gram_matrix[layer], name='L_style') / norm_factor # Do we need the 2

# The loss
L = tf.add(alpha * L_content , beta * L_style, name='L')
# The optimizer
print('Optimizer')
learning_rate = tf.train.exponential_decay(learning_rate=2.0, global_step=global_step, decay_steps=100, decay_rate=0.94, staircase=True, name='learning_rate')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(L, global_step=global_step, var_list=[gen_img], name='train_step')


# The optimizer has variables that require initialization as well
# --------------------------------------------- LAUNCH -----------------------------------------------------------------
sess.run(tf.global_variables_initializer())
for i in range(num_iters):

    aux = sess.run(gen_img_responses)
    gen_image_val = sess.run(gen_img)
    if i%50 == 0:
        save_image(gen_image_val, i, 'C:\\Users\\Serlopal\\Documents\\GitHub\\CNN_Style_Transfer\\style_transfer\\gen_images')
    print("L_content, L_style, L_total:", sess.run(L_content), sess.run(L_style), sess.run(L))
    print("Iter:", i)
    sess.run(train_step)

writer = tf.summary.FileWriter('C:\\Users\\Serlopal\\Documents\\GitHub\\CNN_Style_Transfer\\logs', graph=tf.get_default_graph())
