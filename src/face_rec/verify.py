# -*- coding: utf-8 -*-

import sys
import numpy as np
import face_recognition


# 返回图片中所有人脸的特征
def get_features(filename):
    # extract faces
    image = face_recognition.load_image_file(filename)
    face_bounding_boxes = face_recognition.face_locations(image)
    if len(face_bounding_boxes)==0:
        return [], []

    features = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes, num_jitters=1)

    return features, face_bounding_boxes

