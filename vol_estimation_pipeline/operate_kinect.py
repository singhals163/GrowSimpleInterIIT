import freenect
import cv2
import numpy as np

def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

def get_depth():
    array,_ = freenect.sync_get_depth()
    metric_distance = find_distance(array)
    array1 = (array).astype(np.uint8)
    return metric_distance, array1, array


def find_distance(depth_map):
    estimate_depth =  0.1236 * np.tan((depth_map/ 2842.5) + 1.1863) 
    return estimate_depth