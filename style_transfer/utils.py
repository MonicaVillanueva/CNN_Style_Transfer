from scipy.misc import imsave
from datetime import datetime as dt
import numpy as np
import os

def save_image(im, iteration, out_dir):
    img = im.copy()
    img = np.clip(img, 0, 255).astype(np.uint8) # img[0, ...] ???
    nowtime = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    imsave("{}\pneural_art_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)

