import numpy as np


def error(val1, val2):
    return (abs(val1-val2))

def get_erroneous_objects():
    data = np.loadtxt("boxes_data.txt", dtype = {'names':('sku_id', 'area', 'height', 'volume', 'weight'), 'formats':('S30', np.float32, np.float32, np.float32, np.float32)})
    # boxes = [('afsf', 4924, 47, 4891, 668), ('cfsf', 41, 41, 45124, 79876), ('afsf', 411, 414, 414, 4.35), ('cfsf', 4415, 412, 45167, 67897), ('afsf', 411, 414, 415, 4.5), ('afsf', 411, 414, 420, 4.56), ('afsf', 411, 414, 3000, 4.76)]
    print(data)

    boxes = np.sort(data, order = ['sku_id', 'area', 'height', 'weight', 'volume'])

    print(boxes)

    i = 0
    j = 0
    outliers = []
    while i < len(boxes):
        while j < len(boxes) and boxes[i][0] == boxes[j][0]:
            j+=1
        j-=1
        # print(j)
        mid = (i+j)//2
        for k in range(i, j+1):
            if error(boxes[mid][1], boxes[k][1]) > 10 or error(boxes[mid][2], boxes[k][2]) > 3 or error(boxes[mid][4], boxes[k][4]) > 0.5:
                outliers.append(boxes[k])
        i = j+1
        j = j+1
    print(outliers)
    return outliers