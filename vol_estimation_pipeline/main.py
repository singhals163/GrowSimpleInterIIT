import cv2
import calculate_volume1
import operate_kinect
import ml_classification


########## save initial depth map ############
while 1:
    metric_distance_initial, depth_image, raw_depth = operate_kinect.get_depth()
    cv2.imshow('depth image', depth_image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

flag = 0
object_id = 0



while 1:
    ########### capture frame with object #############
    while 1:
        metric_distance, depth_image, raw_depth = operate_kinect.get_depth()
        cv2.imshow("depth image", depth_image)
        area, object_height, segmented_depth_map = calculate_volume1.find_dimensions(depth_image, metric_distance, metric_distance_initial)
        cv2.imshow("segmented depth map", segmented_depth_map)
        k = cv2.waitKey(5) & 0xFF
        if k == 113:
            flag = 1
            break
        elif k == 114:
            break
    cv2.destroyAllWindows()


    if flag:
        break
    # find dimensions
    # find object type
    object_id = ml_classification.classify_shape(segmented_depth_map)
    
    # for cuboid
    if object_id == 1:
        calculate_volume1.find_cuboid_volume(area, object_height)
    # for cylinder side view
    elif object_id == 2:
        calculate_volume1.find_cylinder_volume(area, object_height)

    # for prism side view
    elif object_id == 3:
        calculate_volume1.find_prism_volume(area, object_height)

    # for pyramid
    # elif object_id == 4:
    #     calculate_volume1.find_pyramid_volume(area, object_height)
    
    # for sphere
    elif object_id == 4:
        calculate_volume1.find_sphere_volume(area, object_height)

    else: 
        object_id = 0