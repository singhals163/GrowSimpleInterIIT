# image segmenting and filtering for getting the object shape
import cv2
import numpy as np

def object_mask(metric_distance_initial,metric_distance):

    # finding depth distance
    diff_distance = metric_distance_initial-metric_distance
    (H,W) = np.shape(diff_distance)
    mask = np.zeros((H,W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if diff_distance[i][j] < 0.006:
                continue
            if diff_distance[i][j] != 0:
                mask[i][j] = 255
    
    # opening(7,7)
    kernel_erode = (7,7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_erode)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    # contours
    contours, hierarchy = cv2.findContours( opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_image = np.zeros(np.shape(diff_distance), dtype=np.uint8)
    cv2.drawContours(original_image, contours, -1, 255, thickness=cv2.FILLED)
    

    contours, heirarchy = cv2.findContours(original_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = []
    final_image = np.zeros(np.shape(diff_distance), dtype=np.uint8)
    area = 0
    if len(contours) != 0:
        c.append(max(contours, key=cv2.contourArea))

        final_image = cv2.drawContours(final_image, c, -1, 255, thickness=cv2.FILLED)
        final_image = cv2.erode(final_image, kernel=kernel_erode)
        area = cv2.contourArea(c[0])

    return final_image, area