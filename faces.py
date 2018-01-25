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
    theta = grad_descent(f, df, x, y, theta_init, 0.00001, 50000)

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

        cost_fn += (prediction)**2

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

        cost_fn += (prediction)**2

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
    # Actors for training and validation set
    act_train = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    act_train_gender = {'Lorraine Bracco': 'female', 
                        'Angie Harmon': 'female', 
                        'Alec Baldwin': 'male', 
                        'Bill Hader': 'male', 
                        'Steve Carell': 'male',
                        'Peri Gilpin': 'female'}

    # Actors for testing set
    act_test = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']
    act_test_gender = {'Daniel Radcliffe': 'male', 
                       'Gerard Butler': 'male', 
                       'Michael Vartan': 'male', 
                       'Kristin Chenoweth': 'female', 
                       'Fran Drescher': 'female', 
                       'America Ferrera': 'female'}

    # Training size (images per actor)
    training_sizes = [10, 50, 65, 100, 120]

    training_sets, validation_sets, testing_sets = {}, {}, {}

    # Build training sets, validation sets and testing sets
    for a in act_train:
        a_name = a.split()[1].lower()
        training_set, validation_set, _ = build_sets(a_name)
        training_sets[a] = training_set
        validation_sets[a] = validation_set

    for a in act_test:
        a_name = a.split()[1].lower()
        training_set, validation_set, testing_set = build_sets(a_name)
        # Concatenate all list to make a bigger testing set
        testing_sets[a] = training_set + validation_set + testing_set

    # Results for correspoding training sizes
    result_training_set, result_validation_set, result_testing_set = [], [], []

    for training_size in training_sizes:
        # Train 
        total_training_examples = 0
        for a in act_train:
            total_training_examples += min(len(training_sets[a]), training_size)

        x = np.zeros((total_training_examples, 1024))
        y = np.zeros(total_training_examples)
	
	print("Training Size: "+str(total_training_examples))	

        i = 0

        for a in act_train:
            for training_example in range(training_size):
                if training_example >= len(training_sets[a]): break

                tr_img = imread("cropped/"+training_sets[a][training_example])
                tr_img = rgb2gray(tr_img)
                x[i] = reshape(np.ndarray.flatten(tr_img), [1, 1024])
                if act_train_gender[a] == 'male': y[i] = 1
                elif act_train_gender[a] == 'female': y[i] = 0
                i += 1

        theta_init = np.zeros(1025)
        theta = grad_descent(f, df, x, y, theta_init, 0.000001, 50000)

        # Performance on training set
        correct, total = 0, 0
        for a in act_train:
            for training_example in range(training_size):
                if training_example >= len(training_sets[a]): break
                
                tr_img = imread("cropped/"+training_sets[a][training_example])
                tr_img = rgb2gray(tr_img)
                tr_img = reshape(np.ndarray.flatten(tr_img), [1, 1024])
                tr_img = np.insert(tr_img, 0, 1)

                prediction = dot(theta.T, tr_img)

                if act_train_gender[a] == 'male':
                    if linalg.norm(prediction) > 0.5: correct += 1
                elif act_train_gender[a] == 'female':
                    if linalg.norm(prediction) < 0.5: correct += 1

                total += 1

	print(total)
        result_training_set.append(correct/float(total))

        # Performance on validation set
        correct, total = 0, 0
        for a in act_train:
            for validation_example in validation_sets[a]:
                v_img = imread("cropped/"+validation_example)
                v_img = rgb2gray(v_img)
                v_img = reshape(np.ndarray.flatten(v_img), [1, 1024])
                v_img = np.insert(v_img, 0, 1)

                prediction = dot(theta.T, v_img)

                if act_train_gender[a] == 'male':
                    if linalg.norm(prediction) > 0.5: correct += 1
                elif act_train_gender[a] == 'female':
                    if linalg.norm(prediction) < 0.5: correct += 1

                total += 1

        result_validation_set.append(correct/float(total))

        # Performance on testing set
        correct, total = 0, 0
        for a in act_test:
            for testing_example in testing_sets[a]:
                t_img = imread("cropped/"+testing_example)
                t_img = rgb2gray(t_img)
                t_img = reshape(np.ndarray.flatten(t_img), [1, 1024])
                t_img = np.insert(t_img, 0, 1)

                prediction = dot(theta.T, t_img)

                if act_test_gender[a] == 'male':
                    if linalg.norm(prediction) > 0.5: correct += 1
                elif act_test_gender[a] == 'female':
                    if linalg.norm(prediction) < 0.5: correct += 1

                total += 1

        result_testing_set.append(correct/float(total))

    print(result_training_set)
    print(result_validation_set)
    print(result_testing_set)

################################################################################
# Part 6
################################################################################
def part6():
    pass

################################################################################
# Part 7
################################################################################
def part7():
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    # Build training, validation and testing sets
    training_sets, validation_sets, testing_sets = {}, {}, {}

    # Equalize each actor's training set to the least examples for any actor
    training_examples_per_actor = 200
    for a in act:
        a_name = a.split()[1].lower()
        training_set, validation_set, testing_set = build_sets(a_name)

        training_examples_per_actor = min(training_examples_per_actor, len(training_set))

        training_sets[a] = training_set
        validation_sets[a] = validation_set
        testing_sets[a] = testing_set

    # Equalize training sets
    for a in act:
        training_sets[a] = training_sets[a][:training_examples_per_actor]

    # Train
    x = np.zeros((training_examples_per_actor * len(act), 1024))
    y = np.zeros((training_examples_per_actor * len(act), len(act)))

    i, a_i = 0

    for a in act:
        for tr in training_sets[a]:
            tr_img = imread("cropped/"+tr)
            tr_img = rgb2gray(tr_img)
            
            x[i] = reshape(np.ndarray.flatten(tr_img), [1, 1024])

            y[i][a_i] = 1

            i += 1

        a_i += 1

    theta_init = np.zeros(1025)
    theta = grad_descent_muticlass(f_multiclass, df_multiclass, x, y, theta_init, 0.000001, 5000)

    # Performance on training set
    correct, total = 0, 0

    a_i = 0
    for a in act:
        for tr in training_sets[a]:
            tr_img = imread("cropped/"+tr)
            tr_img = rgb2gray(tr_img)
            tr_img = reshape(np.ndarray.flatten(tr_img), [1, 1024])
            tr_img.insert(tr_img, 0, 1)

            prediction = dot(theta, tr_img)
            prediction = np.argmax(prediction)

            if prediction == a_i: correct += 1

            total += 1

        a_i += 1

    print("Training Set Performance = "+str(correct)+"/"+str(total))

    # Performance on validation set
    correct, total = 0, 0

    a_i = 0
    for a in act:
        for v in validation_sets[a]:
            v_img = imread("cropped/"+v)
            v_img = rgb2gray(v_img)
            v_img = reshape(np.ndarray.flatten(v_img), [1, 1024])
            v_img.insert(v_img, 0, 1)

            prediction = dot(theta, v_img)
            prediction = np.argmax(prediction)

            if prediction == a_i: correct += 1

            total += 1

        a_i += 1

    print("Validation Set Performance = "+str(correct)+"/"+str(total))

################################################################################
# Part 8
################################################################################
def part8():
    pass

################################################################################
# Function Calls
################################################################################

# part1()
# part2()
# part3()
# part4()
# part5()
# part6()
part7()
# part8()
