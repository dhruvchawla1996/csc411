'''Download
   Images for each actor specified in the act list
   And store original images in uncropped/ 
   And cropped 32x32 (according to bounding box) grayscale images in cropped/
   
   Requires: Empty folders uncropped/ and cropped/ and faces_subtext.txt containing dataset from FaceScrub
'''

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

from rgb2gray import rgb2gray

################################################################################
# Timeout Function
################################################################################
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    Manages download by aborting download if it's taking too long'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work
def get_and_crop_images(act):
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("faces_subset.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 45)

                crop_bbox = line.split()[5].split(',')

                # Convert uncropped image to cropped 32x32 grayscale
                if not os.path.isfile("uncropped/"+filename):
                    continue

                try:
                    rgb_img = imread("uncropped/"+filename)
                    grayscale_img = rgb2gray(rgb_img)
                    cropped_img = grayscale_img[int(crop_bbox[1]):int(crop_bbox[3]), int(crop_bbox[0]):int(crop_bbox[2])]
                    resized_img = imresize(cropped_img, (32, 32))
                    imsave("cropped/"+filename, resized_img, cmap = plt.cm.gray)

                except Exception as e:
                    print(str(e))

                print filename
                i += 1
