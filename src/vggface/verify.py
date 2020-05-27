# -*- coding: utf-8 -*-

import sys

if __name__ == '__main__':
    if len(sys.argv)<3:
        print("usage: python3 %s <img1> <img2>" % sys.argv[0])
        sys.exit(2)


# face verification with the VGGFace2 model
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from .keras_vggface.vggface import VGGFace
from .keras_vggface.utils import preprocess_input

import face_recognition


# 装入识别模型 # pooling: None, avg or max # model: vgg16, senet50, resnet50
model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') 


# 从照片中获取人脸数据，返回所有能识别的人脸
def extract_face(filename, required_size=(224, 224)):
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
        face_array = np.asarray(image, 'float32')
        face_list.append(face_array)

        # show face
        #from PIL import ImageDraw
        #draw = ImageDraw.Draw(image)
        #del draw
        #image.show()

    return face_list, face_bounding_boxes


def load_face(filename, required_size=(224, 224)):
    img = image.load_img(filename, target_size=required_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


# 返回图片中所有人脸的特征
def get_features(filename):
    # extract faces
    faces, face_boxs = extract_face(filename)
    if len(faces) == 0:
        return [], []
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # perform prediction
    yhat = model.predict(samples)
    yhat2 = yhat / np.linalg.norm(yhat)
    return yhat2, face_boxs


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


if __name__ == '__main__':

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    feature1, _ = get_features(filename1)
    feature2, _ = get_features(filename2)

    if len(feature1)>0 and len(feature2)>0:
        is_match(feature1[0], feature2[0])
    else:
        print('fail to get features.')
