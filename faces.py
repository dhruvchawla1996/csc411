'''CSC411: Project 1
    Dhruv Chawla'''

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
    # Actors for training and validation set
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    #get_and_crop_images(act)

    # Actors for testing set (part 5)
    act_test = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']

    get_and_crop_images(act_test)

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
# def part5():
#     # Actors for training and validation set
#     act_train = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
#     act_train_gender = ['male', 'male', 'male', 'female', 'female', 'female']

#     # Actors for testing set
#     act_test = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']
#     act_test_gender = ['male', 'male', 'male', 'female', 'female', 'female']

#     # Training size (images per actor)
#     training_sizes = [5, 10, 20, 40, 50, 65]

#     training_sets, validation_sets, testing_sets = [], [], []

#     # Build training sets, validation sets and testing sets
#     for a in act_train:
#         a_name = a.split()[1].lower()
#         training_set, validation_set, _ = build_sets(a_name)
#         training_sets.append(_training_set)
#         validation_sets.append(validation_set)

#     for a in act_test:
#         a_name = a.split()[1].lower()
#         _1, _2, testing_set = build_sets(a_name)
#         testing_sets.append(testing_set)

#     # Results for correspoding training sizes
#     result_training_set, result_validation_set, result_testing_set = [], [], []

#     for training_size in training_sizes:
#         # Train



part1()