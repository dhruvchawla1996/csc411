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
from numpy import linalg

from get_data import *
from rgb2gray import rgb2gray
from build_sets import *
from calculus import *

################################################################################
# Part 1
################################################################################
def part1():
    get_and_crop_images()

################################################################################
# Part 2
################################################################################
def part2():
    a = 'Lorraine Bracco'
    a_name = a.split()[1].lower()
    training_set, validation_set, testing_set = build_sets(a_name)

################################################################################
# Part 3
################################################################################
def part3():
    actor_0, actor_1 = 'Alec Baldwin', 'Steve Carell'

    a_0_name = actor_0.split()[1].lower()
    training_set_0, validation_set_0, testing_set_0 = build_sets(a_0_name)

    a_1_name = actor_1.split()[1].lower()
    training_set_1, validation_set_1, testing_set_1 = build_sets(a_1_name)

    total_training_examples = len(training_set_0) + len(training_set_1)

    # Train 
    x = np.zeros((total_training_examples, 1024))
    y = np.zeros(total_training_examples)

    i = 0

    for tr_0 in training_set_0:
        tr_img = imread("cropped/"+tr_0)
        tr_img = rgb2gray(tr_img)
        x[i] = reshape(np.ndarray.flatten(tr_img), [1, 1024])
        y[i] = 1
        i += 1

    for tr_1 in training_set_1:
        tr_img = imread("cropped/"+tr_1)
        tr_img = rgb2gray(tr_img)
        x[i] = reshape(np.ndarray.flatten(tr_img), [1, 1024])
        y[i] = 0
        i += 1

    theta_init = np.zeros(1025)
    theta = grad_descent(f, df, x, y, theta_init, 0.00001)

    # Performance on Training Set
    correct, total, cost_fn = 0, 0, 0

    for tr_0 in training_set_0:
        tr_img = imread("cropped/"+tr_0)
        tr_img = rgb2gray(tr_img)
        tr_img = reshape(np.ndarray.flatten(tr_img), [1, 1024])
        tr_img = np.insert(tr_img, 0, 1)

        prediction = dot(theta.T, tr_img)

        cost_fn += (1 - prediction)**2

        if linalg.norm(prediction) > 0.5: correct += 1
        total += 1

    for tr_1 in training_set_1:
        tr_img = imread("cropped/"+tr_1)
        tr_img = rgb2gray(tr_img)
        tr_img = reshape(np.ndarray.flatten(tr_img), [1, 1024])
        tr_img = np.insert(tr_img, 0, 1)

        prediction = dot(theta.T, tr_img)

        cost_fn += (1 - prediction)**2

        if linalg.norm(prediction) < 0.5: correct += 1
        total += 1

    print("Training Set Performance = "+str(correct)+"/"+str(total))
    print("Cost Function value for training set is "+str(cost_fn))

    # Performance on Validation Set
    correct, total, cost_fn = 0, 0, 0

    for v_0 in validation_set_0:
        v_img = imread("cropped/"+v_0)
        v_img = rgb2gray(v_img)
        v_img = reshape(np.ndarray.flatten(v_img), [1, 1024])
        v_img = np.insert(v_img, 0, 1)

        prediction = dot(theta.T, v_img)

        cost_fn += (1 - prediction)**2

        if linalg.norm(prediction) > 0.5: correct += 1
        total += 1

    for v_1 in validation_set_1:
        v_img = imread("cropped/"+v_1)
        v_img = rgb2gray(v_img)
        v_img = reshape(np.ndarray.flatten(v_img), [1, 1024])
        v_img = np.insert(v_img, 0, 1)

        prediction = dot(theta.T, v_img)

        cost_fn += (1 - prediction)**2

        if linalg.norm(prediction) < 0.5: correct += 1
        total += 1

    print("Validation Set Performance = "+str(correct)+"/"+str(total))
    print("Cost Function value for validation set is "+str(cost_fn))

################################################################################
# Part 4
################################################################################
def part4():
    pass

################################################################################
# Part 5
################################################################################
def part5():
    pass

part3()