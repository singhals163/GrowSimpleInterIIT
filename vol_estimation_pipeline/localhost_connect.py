from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
import base64
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
import numpy as np
import os
import ml_classifcation as predict2
import open3d as o3d
from importlib import reload  # reload
import cv2
import calculate_volume1
import operate_kinect
# import predict

# from predict import *


# import stream
@app.route("/")
def index():
    return render_template("index.html")

a = 0
metric_distance_initial = None

r = 0


@socketio.on("config")
def config():
    global r
    r = 1


@socketio.on("initialise")
def initialise():
    global metric_distance_initial, r
    while 1:
        q = "2"
        metric_distance_initial, depth_image, raw_depth = operate_kinect.get_depth()
        rgb= operate_kinect.get_video()
        imgencode = cv2.imencode('.jpg', depth_image,[cv2.IMWRITE_JPEG_QUALITY,40])[1]
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData = b64_src + stringData
        imgencode1 = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]
        stringData1 = base64.b64encode(imgencode1).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData1 = b64_src + stringData1
        emit('response_back', (stringData,stringData1))

        if r == 1:
            q = "3"
            print("ok")
            r = 0
            break
    print("ok2")


metric_distance = None
depth_image = None
raw_depth = None
r2 = 0
object_id = 0
pred=1
import depth_transformation
@socketio.on("toggled")
def change():
    global pred
    if pred==1:
        pred=0
    if pred == 0:
        pred=1
@socketio.on("calc_vol")
def calc_vol():
    global r2
    r2 = 1
    vol = 0
    # send(69,broadcast=True)

    global object_id, depth_image, metric_distance, metric_distance_initial, raw_depth

    area, object_height, segmented_depth_map = calculate_volume1.find_dimensions(depth_image, metric_distance,
                                                                                 metric_distance_initial)
    xyz, uv = depth_transformation.depth2xyzuv(raw_depth)
    print(np.asarray(xyz).shape)
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("678.ply",pcd)
    object_id=predict2.classify_shape(segmented_depth_map)
    #object_id = predict.pred(np.asarray(xyz), area)

    if object_id==3:
            object_id=0
    if object_id==4:
            object_id=3
    '''        
    if pred==1:
     object_id = predict.pred(np.asarray(xyz),area)
    if pred==0:
        #object_id=predict2.classify_shape(segmented_depth_map)
        object_id = predict.pred(np.asarray(xyz), area)

        if object_id==3:
            object_id=0
        if object_id==4:
            object_id=3
    else:
      object_id = predict.pred(np.asarray(xyz), area)
    '''  
    if object_id == 1:
        vol = calculate_volume1.find_cuboid_volume(area, object_height)
        pass
    # for cylinder side view
    elif object_id == 2:
        vol = calculate_volume1.find_cylinder_volume(area, object_height)
    # for prism side view
    elif object_id == 0:
        vol = calculate_volume1.find_prism_volume(area, object_height)
    # for sphere
    elif object_id == 3:
        vol = calculate_volume1.find_sphere_volume(area, object_height)

    else:
        object_id = 0
   
    # send("69", broadcast=True)
    vol=vol//1
    vol=int(abs(vol))
    vol=str(vol)+"cm3"
    print(vol)
    socketio.emit("qwer", vol)


@socketio.on("init2")
def init2():
    global metric_distance, depth_image, raw_depth, r2
    while 1:

        metric_distance, depth_image, raw_depth = operate_kinect.get_depth()
        rgb= operate_kinect.get_video()
        if r2 == 1:
            r2 = 0
            break
        imgencode = cv2.imencode('.jpg', depth_image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData = b64_src + stringData
        imgencode1 = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]
        stringData1 = base64.b64encode(imgencode1).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData1 = b64_src + stringData1
        emit('response_back2', (stringData, stringData1))

app.run(debug=False)
