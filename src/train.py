# -*- coding: utf-8 -*-

import os, sys
from settings import ALGORITHM
import knn


'''
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
'''


if __name__ == "__main__":
    if len(sys.argv)<3:
        print("usage: python3 %s <algorithm> <train_data_dir> [model_name]" % sys.argv[0])
        sys.exit(2)

    face_algorithm = sys.argv[1]

    if face_algorithm not in ALGORITHM.keys():
        print('Algorithm not found!')
        sys.exit(2)

    train_data_dir = sys.argv[2]

    if len(sys.argv)>3:
        model_name = sys.argv[3]
    else:
        model_name = 'trained_knn_model'

    # Train the KNN classifier and save it to disk
    print("Training KNN classifier...")
    classifier = knn.train(train_data_dir, 
        model_save_path=model_name + ALGORITHM[face_algorithm]['ext'], 
        n_neighbors=2,
        face_algorithm=face_algorithm)
    print("Training complete!")
