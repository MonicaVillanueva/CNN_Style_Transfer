import tensorflow as tf
from scipy.misc import imread, imresize
import numpy as np


def read_image(path, height, width):
    img = imread(path, mode='RGB')
    img = imresize(img, (height, width))

    return img


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


def create_white_noise(mean, std, height, width):
    return np.random.normal(mean, std, size=(height, width, 3)).astype('float32')


def compute_content_loss(sess, resp_orig, resp_noise):
    """
    Equation 1 from paper
    """
    loss = []
    for layer in range(len(resp_orig)):
        loss_lay = 0.5 * tf.reduce_sum(tf.pow(resp_orig[layer] - resp_noise[layer], 2))
        loss.append(sess.run(loss_lay))
    return loss


def compute_style_loss(sess, feat_maps, gram_orig, gram_noise):
    """
    Equation 4 and 5 from paper
    """

    contributions = []
    for layer in range(len(feat_maps)):
        N = feat_maps[layer].shape[3]                                   # distinct filters
        M = feat_maps[layer].shape[1] * feat_maps[layer].shape[2]       # size of each feature map

        contributions.append(sess.run((1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(gram_orig[layer] - gram_noise[layer], 2))))

    loss = tf.reduce_sum(style_W * contributions) #TODO
    return sess.run(loss)