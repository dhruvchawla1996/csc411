'''CSC411: Project 1
    Dhruv Chawla and Sabrina Lokman'''

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

from get_data import *
from build_sets import *

# Part 1
def part1():
    get_and_crop_images()

# Part 2:
def part2():
	a = 'Lorraine Bracco'
	a_name = a.split()[1].lower()
	training_set, validation_set, testing_set = build_sets(a_name)
