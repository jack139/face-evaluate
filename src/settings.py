# -*- coding: utf-8 -*-

from importlib import import_module

ALGORITHM = {
    'rec' : { 'distance_threshold' : 0.6, 'ext' : '.rec.clf', 'module' : 'face_rec.verify' },
    'vgg'  : { 'distance_threshold' : 0.8, 'ext' : '.vgg.clf', 'module' : 'vggface.verify' },
    'evo'  : { 'distance_threshold' : 1.25, 'ext' : '.evo.clf', 'module' : 'face_evoLVe.verify' },
    'plus'  : { 'distance_threshold' : 1.4, 'ext' : '.plus.clf', 'module' : 'verify_plus' },
}

# 并行算法设置
algorithm_settings = {
    1 : [ 'vgg', 'data/model/train2_senet50.vgg.clf' ], # 优先返回
    2 : [ 'evo', 'data/model/train2_5.evo.clf' ],
    #1 : [ 'rec', 'data/model/train.rec.clf' ],
    #2 : [ 'vgg', 'data/model/train_senet50.vgg.clf' ],
}

def import_verify(face_algorithm):
    module = import_module(ALGORITHM[face_algorithm]['module'])
    #print('imported: ', module)
    return module
