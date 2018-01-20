# Imports

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

def build_sets(actor):
    '''Return three lists of randomized image names 
    in cropped/ folder that match actor name
    
    Training Set - At least 67 image names (Screw Peri Gilpin)
    Validation Set - 10 image names 
    Testing Set - 10 image names
    
    Takes in name as lowercase last name (ex: gilpin)

    Assumption: cropped/ folder is populated with images from get_and_crop_images
    '''
    # Make a list of images for the actor
    image_list = []

    for f in os.listdir("cropped"):
        if actor in f:
            image_list.append(f)

    # Shuffle
    np.random.seed(20)
    np.random.shuffle(image_list)

    # 10 images for testing and validation set each and the rest in training set
    testing_set = image_list[0:10]
    validation_set = image_list[10:20]
    training_set = image_list[20:]

    return training_set, validation_set, testing_set