import tensorflow as tf
import os
import numpy as np
from utils import save_image
from VGG import vgg16st as net
from scipy.misc import imread, imresize


## Constants
PIC_SIZE = 448
CHANNELS = 3

# Parameters
INI_ETA = 10.0
DECAY = 0.94
STEPS = 100
NUM_ITERS = 51
alpha = tf.constant(1.0, name='alpha')
beta = tf.constant(10000.0, name='beta')
noisy = False  # Flag for initial image (True = Random noisy Image, False = Content image)


## Paths
model_path = os.path.join('VGG', 'vgg16_weights.npz')
original_images_path = os.path.join('Dataset', 'video')
photo_path = os.path.join(original_images_path, 'cat_012.jpg')
paint_path = os.path.join(original_images_path, 'klimt.jpg')
gen_images_path = os.path.join(os.getcwd(), 'gen_images')
logs_path = os.path.join(os.getcwd(), 'logs')


## Model
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, PIC_SIZE, PIC_SIZE, CHANNELS])
vgg = net.vgg16(imgs, model_path, sess)


## Input images
content_image = imread(photo_path, mode='RGB')
content_image = imresize(content_image, (PIC_SIZE, PIC_SIZE))

style_image = imread(paint_path, mode='RGB')
style_image = imresize(style_image, (PIC_SIZE, PIC_SIZE))


## Load images into the net
print("Get content responses")
content_constant = tf.constant(content_image, dtype=tf.float32, name="content_constant")
vgg_content = net.vgg16(content_constant, model_path, sess)
content_responses = [vgg_content.conv1_1, vgg_content.conv2_1, vgg_content.conv3_1, vgg_content.conv4_1, vgg_content.conv5_1]

print("Get style responses")
style_constant = tf.constant(style_image, dtype=tf.float32)
vgg_style = net.vgg16(style_constant, model_path, sess)
style_responses = [vgg_style.conv1_1, vgg_style.conv2_1, vgg_style.conv3_1, vgg_style.conv4_1, vgg_style.conv5_1]

print("Compute Gram Matrix")# TODO: generalize to use any number of layers
style_gram_matrix = []
for layer in range(len(style_responses)):
    # Vectorize response per filter
    num_filters = style_responses[layer].get_shape().as_list()[3]
    vec_shape = [-1, num_filters]    # [-1] flattens 3D to 2D matrix
    vec_resp = tf.reshape(style_responses[layer], vec_shape)  # vectorize responses (to multiply)
    # Compute Gram Matrix as correlation one layer with itself
    corr = tf.matmul(tf.transpose(vec_resp), vec_resp, name='corr')
    style_gram_matrix.append(corr)

# Build graph
L_content = tf.Variable(0.0, name="L_content_var")
L_style = tf.Variable(0.0, name="L_style")

# Initialize noisy image
if noisy == True:
    gen_img = tf.Variable(tf.truncated_normal(content_image.shape, stddev=20, dtype=tf.float32),
                          dtype=tf.float32, trainable=True, name='gen_img')
else:
    gen_img = tf.Variable(content_image, dtype=tf.float32, name="gen_img")
vgg_aux = net.vgg16(gen_img, model_path, sess)
gen_img_responses = [vgg_aux.conv1_1, vgg_aux.conv2_1, vgg_aux.conv3_1, vgg_aux.conv4_1, vgg_aux.conv5_1]


print('After image')
for layer in range(len(gen_img_responses)):
    # Content loss
    if layer == 4:
        L_content += tf.nn.l2_loss(content_responses[layer] - gen_img_responses[layer], name='L_content')

    # Style
    num_filters = style_responses[layer].get_shape().as_list()[3]
    vec_shape = [-1, num_filters]  # [-1] flattens 3D into 2D matrix
    vec_resp = tf.reshape(gen_img_responses[layer], vec_shape, name='vec_resp')  # vectorize responses (to multiply)

    # Compute Gram Matrix as correlation one layer with itself
    corr = tf.matmul(tf.transpose(vec_resp), vec_resp, name='corr')
    N = np.prod(content_responses[layer].get_shape().as_list()).astype(np.float32)
    # N is the product of the 3 dimensions of the responses, because we want N = number_filters * size(feature_map)
    norm_factor = tf.constant((2 * (N**2)) * len(content_responses), name='norm_factor', dtype='float32')
    L_style += tf.divide(tf.nn.l2_loss(corr - style_gram_matrix[layer]), norm_factor, name='L_style')

# The loss
L = tf.add(alpha * L_content , beta * L_style, name='L')
# The optimizer
print('Optimizer')
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(learning_rate=INI_ETA, global_step=global_step, decay_steps=STEPS,
                                           decay_rate=DECAY, staircase=True, name='learning_rate')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(L, global_step=global_step, var_list=[gen_img],
                                                            name='train_step')


# The optimizer has variables that require initialization as well
# --------------------------------------------- LAUNCH -----------------------------------------------------------------
sess.run(tf.global_variables_initializer())
for i in range(NUM_ITERS):

    gen_image_val = sess.run(gen_img)
    if i == 20 or i == 50:
        save_image(gen_image_val, i, gen_images_path)
    print("L_content, L_style, L_total:", sess.run(L_content), sess.run(L_style), sess.run(L))
    print("Iter:", i)
    sess.run(train_step)

# Write graph
# if not os.path.exists(logs_path):
#     os.mkdir(logs_path)
# writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

