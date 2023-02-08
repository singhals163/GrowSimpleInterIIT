# getting the top view by applying flattening algorithm

import cv2
import numpy as np

max_distance = 1.0

def get_top_view(depth_frame, distance_matrix, mask):
    (H,W) = np.shape(depth_frame)

    top_view_image = np.zeros(np.shape(depth_frame), dtype = np.uint8)
    new_mask = np.zeros(np.shape(depth_frame), dtype = np.uint8)

    obj_height = 50.0
    h = 0
    k = 0
    for i in range(H):
        for j in range(W):
            if mask[i][j] > 0 and distance_matrix[i][j] > 0:
                if distance_matrix[i][j] < obj_height:
                    h = i
                    k = j
                    obj_height = min(obj_height, distance_matrix[i][j])
                j_final = int((j-W//2)*distance_matrix[i][j] + W//2)
                i_final = int(H//2 - (H//2 - i) * distance_matrix[i][j])
                if i_final >= 0 and i_final < 480 and j_final >= 0 and j_final < 640:
                    top_view_image[i_final][j_final] = depth_frame[i][j]
                    new_mask[i_final][j_final] = 255

    # opening(7,7)
    # cv2.imshow("new mask", new_mask)
    # kernel_size = (3,3)
    # kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # opening = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel_opening)
    # cv2.imshow("opening", opening)
    kernel_size = (15,15)
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel_closing)
    # cv2.imshow("closing", closing)

    # top_view_image = np.uint8(top_view_image * (closing/255))

    contours, heirarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = []
    final_image = np.zeros(np.shape(depth_frame), dtype=np.uint8)
    area = 0
    if len(contours) != 0:
        c.append(max(contours, key=cv2.contourArea))

        final_image = cv2.drawContours(final_image, c, -1, 255, thickness=cv2.FILLED)
        # final_image = cv2.erode(final_image, kernel=kernel_erode)
        area = cv2.contourArea(c[0])
    # cv2.imshow('MORPHED OPEN IMAGE', opening)
    
    return final_image, closing, area, obj_height, h, k
