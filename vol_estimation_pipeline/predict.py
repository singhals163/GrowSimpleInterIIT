import numpy as np
import os
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Lambda
from importlib import reload # reload

reload(tf)
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import open3d as o3d
#tf.config.experimental_run_functions_eagerly(True)
k=5
tf.compat.v1.disable_eager_execution()
@tf.keras.utils.register_keras_serializable()
def mat_mul(A, B):
    return tf.matmul(A, B)
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
num_points = 2048
from tensorflow.keras.optimizers import Adam
adam = tf.keras.optimizers.legacy.Adam(lr=0.001, decay=0.5)
o=0.01
input_points = Input(shape=(num_points, 3))
x = Convolution1D(64, 1, activation=keras.layers.LeakyReLU(alpha=o) ,input_shape=(num_points, 3))(input_points)
x = BatchNormalization()(x)
x = Convolution1D(128, 1, activation=keras.layers.LeakyReLU(alpha=o))(x)
x = BatchNormalization()(x)
x = Convolution1D(1024, 1, activation=keras.layers.LeakyReLU(alpha=o))(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=num_points)(x)
x = Dense(512, activation=keras.layers.LeakyReLU(alpha=o))(x)
x = BatchNormalization()(x)
x = Dense(256, activation=keras.layers.LeakyReLU(alpha=o))(x)
x = BatchNormalization()(x)
x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
input_T = Reshape((3, 3))(x)

# Forward net
g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation=keras.layers.LeakyReLU(alpha=o))(g)
g = BatchNormalization()(g)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation=keras.layers.LeakyReLU(alpha=o))(g)
g = BatchNormalization()(g)

# Feature Transform net
f = Convolution1D(64, 1, activation=keras.layers.LeakyReLU(alpha=o))(g)
f = BatchNormalization()(f)
f = Convolution1D(128, 1, activation=keras.layers.LeakyReLU(alpha=o))(f)
f = BatchNormalization()(f)
f = Convolution1D(1024, 1, activation=keras.layers.LeakyReLU(alpha=o))(f)
f = BatchNormalization()(f)
f = MaxPooling1D(pool_size=num_points)(f)
f = Dense(512, activation=keras.layers.LeakyReLU(alpha=o))(f)
f = BatchNormalization()(f)
f = Dense(256, activation=keras.layers.LeakyReLU(alpha=o))(f)
f = BatchNormalization()(f)
f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
feature_T = Reshape((64, 64))(f)

# Forward net
g = Lambda(mat_mul, arguments={'B': feature_T})(g)
g = Convolution1D(64, 1, activation=keras.layers.LeakyReLU(alpha=o))(g)
g = BatchNormalization()(g)
g = Convolution1D(128, 1, activation=keras.layers.LeakyReLU(alpha=o))(g)
g = BatchNormalization()(g)
g = Convolution1D(1024, 1, activation=keras.layers.LeakyReLU(alpha=o))(g)
g = BatchNormalization()(g)

# Global_feature
global_feature = MaxPooling1D(pool_size=num_points)(g)

# Point_net_classification
c = Dense(512, activation=keras.layers.LeakyReLU(alpha=o))(global_feature)
c = BatchNormalization()(c)
c = Dropout(rate=0.2)(c)
c = Dropout(rate=0.2)(c)
c = Dense(256, activation=keras.layers.LeakyReLU(alpha=o))(c)
c = BatchNormalization()(c)
c = Dropout(rate=0.2)(c)
c = Dense(k, activation='softmax')(c)
prediction = Flatten()(c)

model = Model(inputs=input_points, outputs=prediction)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from keras.models import load_model
graph = tf.compat.v1.get_default_graph()
model.load_weights('classification_weights4.h5')
print("Weights loaded !")
import numpy as np
import pre_proc
from keras import backend as K

import threading
Session = tf.compat.v1.keras.backend.get_session()
__Graph = tf.compat.v1.get_default_graph()

def pred(data, area):
    pcd=o3d.io.read_point_cloud("678.ply")
    data2=np.asarray(pcd.points)
    target=np.asarray(pre_proc.process(area, data2))
    target = target.reshape(-1, num_points, 3)
    print(target.shape)
    with Session.as_default():
        with __Graph.as_default():
         pred = model.predict(target)
    pred = np.squeeze(pred)
    pred = pred.tolist()
    return np.asarray(pred).argmax()
