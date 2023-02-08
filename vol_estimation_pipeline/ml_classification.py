# for ml classification
##############################################################

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, SimpleRNN, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)
from matplotlib import pyplot as plt

#import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


############## load model #############
resnet_model = tf.keras.models.load_model("./object_classifier2.h5")

############## classifying helper function #############
def classify_shape(depth_image):
    depth_image = cv2.merge((depth_image, depth_image, depth_image))
    img = cv2.resize(depth_image,(256,256))
    img=np.expand_dims(img,axis=0)
    y = resnet_model.predict(img)
    max_index = 0
    max_val = 0
    for i in range(np.shape(y)[1]):
        if max_val<y[0][i]:
            max_index = i
            max_val = y[0][i]
    return max_index+1
