import cv2
import numpy as np
import image_segmentation
import top_view
import object_classifcation

# find the dimensions of object
def find_dimensions(depth_image, metric_distance, metric_distance_initial):
    (H,W) = np.shape(depth_image)
    segmented_image, area = image_segmentation.object_mask(metric_distance_initial, metric_distance)
    masked_image = np.uint8(depth_image*(segmented_image/255))
    top_view_image, new_mask, area, obj_height, h, k = top_view.get_top_view(masked_image, metric_distance, segmented_image)
    obj_height1 = (metric_distance_initial[h][k] - obj_height)*100
    area = area*10000/(594.21*591.04)

    # to visualize the outputs
    # cv2.imshow('depth map', depth_image)
    # cv2.imshow('segmented image', segmented_image)
    # cv2.imshow('top view', top_view_image)
    # cv2.imshow('new mask view', new_mask)

    return area, obj_height1, masked_image

# calculate cuboid's volume
def find_cuboid_volume(area, obj_height):
    volume = area*obj_height
    print("cuboid area: ", area, " height: ", obj_height, " volume: ", volume)

# calcualte sphere's volume
def find_sphere_volume(area, obj_height):
    volume = np.pi*(obj_height**3)/6
    print("sphere diameter: ", obj_height, " volume: ", volume)

# calculate cylinder's volume
def find_cylinder_volume(area, obj_height):
    volume = area*obj_height*np.pi/4
    print("cylinder area: ", area, " height: ", obj_height, " volume: ", volume)

# calculate prism's volume
def find_prism_volume(area, obj_height):
    volume = area*obj_height*0.5
    print("prism area: ", area, " height: ", obj_height, " volume: ", volume)

# calculate pyramid's volume
def find_pyramid_volume(area, obj_height):
    volume = area*obj_height/3
    print("pyramid area: ", area, " height: ", obj_height, " volume: ", volume)