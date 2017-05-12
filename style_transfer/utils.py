import tensorflow as tf
from scipy.misc import imread, imresize, imsave
import numpy as np
import os
import matplotlib.pyplot as plt


def read_image(path, height, width, sub_mean):
    img = imread(path, mode='RGB')
    img = imresize(img, (height, width))
    return img - sub_mean


def save_image(img, mean, epoch, out_path=os.getcwd()):
    # If dir does not exit
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Path
    fname = 'result_epoch', epoch, '.jpg'
    fname = ''.join(map(str, fname))
    if out_path != os.getcwd():
        path = os.path.join(out_path, fname)
    else:
        path = fname

    # Image
    img_save = img + mean
    # img_save = np.clip(img_save, 0, 255).astype('uint8')

    # Save
    imsave(path, img_save)



def create_white_noise(mean, std, height, width, sub_mean):
    img = np.random.normal(mean, std, size=(height, width, 3)).astype('float32')
    return img - sub_mean


def compute_gram_matrix(sess, responses):
    """
    Equation 3 from paper
    """

    gram_matrix = []
    for layer in range(len(responses)):
        # Vectorize response per filter
        num_filters = responses[layer].shape[3]
        vec_shape = [-1, num_filters]  # [-1] flattens into 1-D.
        vec_resp = tf.reshape(responses[layer], vec_shape)  # vectorize responses (to multiply)

        # Compute Gram Matrix as correlation one layer with itself
        corr = tf.matmul(tf.transpose(vec_resp), vec_resp)
        gram_matrix.append(sess.run(corr))

    return gram_matrix


def compute_content_loss(sess, resp_orig, resp_noise):
    """
    Equation 1 from paper
    """
    loss = []
    for layer in range(len(resp_orig)):
        loss_lay = 0.5 * tf.reduce_sum(tf.pow(resp_orig[layer] - resp_noise[layer], 2))
        loss.append(sess.run(loss_lay))
    return sum(loss)


def compute_style_loss(sess, feat_maps, gram_orig, gram_noise):
    """
    Equation 4 and 5 from paper
    """

    loss = 0
    for layer in range(len(feat_maps)):
        N = feat_maps[layer].shape[3]                                   # distinct filters
        M = feat_maps[layer].shape[1] * feat_maps[layer].shape[2]       # size of each feature map

        contribution = sess.run((1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(gram_orig[layer] - gram_noise[layer], 2)))
        style_W = 1. / len(feat_maps)
        loss = loss + style_W * contribution

    return loss