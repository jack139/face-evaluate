# -*- coding: utf-8 -*-

# 合并evo和vgg两个特征值

import numpy as np
from vggface import verify as verify1
from face_evoLVe import verify as verify2

# 定位人脸，然后人脸的特征值列表，可能不止一个脸
def get_features(filename):
    encoding_list1, face_boxes1 = verify1.get_features(filename)
    encoding_list2, face_boxes2 = verify2.get_features(filename)

    encoding_list = []
    for i in range(len(face_boxes1)):
        encoding_list.append(np.concatenate((encoding_list1[i], encoding_list2[i]), axis=0))

    return encoding_list, face_boxes1

