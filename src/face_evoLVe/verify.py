# -*- coding: utf-8 -*-

import cv2
from PIL import Image
import numpy as np
#from .backbone.model_irse import IR_50, IR_152, IR_SE_50, IR_SE_152
from .backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import face_recognition
from .extract_feature_v2 import extract_feature

# 当前使用模型的索引，选择数据模型只需要修改这里
CURRENT_MODEL = 5

INPUT_SIZE = [112, 112]
MODEL_BASE = '/tmp/face_model/face.evoLVe.PyTorch/'
MODEL_LIST = [
    ('ir50', 'bh-ir50/backbone_ir50_asia.pth'), # 0
    ('ir50', 'ms1m-ir50/backbone_ir50_ms1m_epoch63.pth'), # 1
    ('ir50', 'ms1m-ir50/backbone_ir50_ms1m_epoch120.pth'), # 2
    ('ir152', 'ms1m-ir152/Backbone_IR_152_Epoch_37_Batch_841528_Time_2019-06-06-02-06_checkpoint.pth'), # 3
    ('ir152', 'ms1m-ir152/Backbone_IR_152_Epoch_59_Batch_1341896_Time_2019-06-14-06-04_checkpoint.pth'), # 4
    ('ir152', 'ms1m-ir152/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'), # 5
    # 用于 HEAD，不能用于 backbone
    #('ir152', 'ms1m-ir152/Head_ArcFace_Epoch_37_Batch_841528_Time_2019-06-06-02-06_checkpoint.pth'), # 6
    #('ir152', 'ms1m-ir152/Head_ArcFace_Epoch_59_Batch_1341896_Time_2019-06-14-06-04_checkpoint.pth'), # 7
    #('ir152', 'ms1m-ir152/Head_ArcFace_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'), # 8
]

if MODEL_LIST[CURRENT_MODEL][0]=='ir152':
    BACKBONE = IR_152(INPUT_SIZE)
else:
    BACKBONE = IR_50(INPUT_SIZE)

MODEL_ROOT = MODEL_BASE+MODEL_LIST[CURRENT_MODEL][1]

print('Model: ', MODEL_LIST[CURRENT_MODEL][0])
print('Model path: ', MODEL_ROOT)


# 从照片中获取人脸数据，返回所有能识别的人脸
def extract_face(filename, required_size=[112, 112]):
    # load image from file
    pixels = face_recognition.load_image_file(filename)
    # extract the bounding box from the first face
    face_bounding_boxes = face_recognition.face_locations(pixels)

    # 可能返回 >0, 多个人脸
    if len(face_bounding_boxes) == 0:
        return [], []

    face_list = []
    for face_box in face_bounding_boxes:
        top, right, bottom, left = face_box
        x1, y1, width, height = left, top, right-left, bottom-top
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        # transfer to opencv image
        open_cv_image = np.array(image) 
        face_list.append(open_cv_image)

        # show face
        #from PIL import ImageDraw
        #draw = ImageDraw.Draw(image)
        #del draw
        #image.show()

    return face_list, face_bounding_boxes


# 定位人脸，然后人脸的特征值列表，可能不止一个脸
def get_features(filename):
    face_list, face_boxes = extract_face(filename, required_size=INPUT_SIZE)
    encoding_list = []
    for face in face_list:
        open_cv_face = face[:, :, ::-1].copy() 

        face_encodings = extract_feature(open_cv_face, BACKBONE, MODEL_ROOT)
        encoding_list.append(face_encodings.numpy()[0]) # torch.tensor to numpy.array

    return encoding_list, face_boxes

# 直接返回特征值
def get_features2(filename):
    img = cv2.imread(filename)
    face_encodings = extract_feature(img, BACKBONE, MODEL_ROOT)
    return face_encodings

# 定位人脸测试
def test(filename):
    face_list, face_boxes = extract_face(filename, required_size=INPUT_SIZE)
    print(face_boxes)

    n=0
    for face in face_list:
        open_cv_face = face[:, :, ::-1].copy() 
        cv2.imwrite(str(n)+'.jpg', open_cv_face)
        n+=1

# 特征值距离
def face_distance(face_encodings, face_to_compare):
    return face_recognition.face_distance(face_encodings, face_to_compare)
