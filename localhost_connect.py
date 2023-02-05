from flask import Flask, render_template
from flask_socketio import SocketIO, send
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Lambda
from importlib import reload # reload
import operate_kinect
import cv2
import calculate_volume1
import operate_kinect
import ml_classification

reload(tf)
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import open3d as o3d
from sklearn import preprocessing as p
#tf.config.experimental_run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()
#@tf.keras.utils.register_keras_serializable()

#from predict import *





@app.route("/")
def index():
    return render_template("index.html")
a=0
metric_distance_initial=None
 
r=0
@socketio.on("config")
def config():
    global r
    r=1
@socketio.on("initialise")
def initialise():
    global metric_distance_initial,r
    while 1:
        q="2"
        if r==1:
            q="3"
            send(q, broadcast= True)
            print("ok")
            r=0
            #cv2.destroyAllWindows()
            break
        metric_distance_initial, depth_image, raw_depth = operate_kinect.get_depth()
        #cv2.imshow("depth image", depth_image)

        #k = cv2.waitKey(1) 
        send(q, broadcast= True)
    print("ok2")
metric_distance=None
depth_image=None
raw_depth=None
r2=0
object_id = 0
@socketio.on("calc_vol")
def calc_vol():
    global r2
    r2=1
    vol=0
    #send(69,broadcast=True)
    global object_id,depth_image, metric_distance, metric_distance_initial, raw_depth
    area, object_height, segmented_depth_map = calculate_volume1.find_dimensions(depth_image, metric_distance, metric_distance_initial)
    object_id = ml_classification.classify_shape(segmented_depth_map)
    if object_id == 1:
        vol=calculate_volume1.find_cuboid_volume(area, object_height)
        pass
    # for cylinder side view
    elif object_id == 2:
        vol=calculate_volume1.find_cylinder_volume(area, object_height)
    # for prism side view
    elif object_id == 3:
        vol=calculate_volume1.find_prism_volume(area, object_height)
    # for pyramid
    # elif object_id == 4:
    #     calculate_volume1.find_pyramid_volume(area, object_height)
    # for sphere
    elif object_id == 4:
        vol=calculate_volume1.find_sphere_volume(area, object_height)

    else: 
        object_id = 0
    #send("69", broadcast=True)
    socketio.emit("qwer", vol)
@socketio.on("init2")
def init2():
    global metric_distance, depth_image, raw_depth, r2
    while 1:
        
        metric_distance, depth_image, raw_depth = operate_kinect.get_depth()
        #cv2.imshow("depth_image", depth_image)
        #print("ok3")
        #k = cv2.waitKey(1) & 0xFF
        #if k == 113:
            
        #    break
        #elif k == 114:
        #    break
        if r2==1:
            #cv2.destroyAllWindows()
            r2=0
            break





app.run(debug=True)