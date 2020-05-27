# -*- coding: utf-8 -*-

# 使用两个算法模型并行识别

import os, sys
import concurrent.futures
from datetime import datetime
from settings import ALGORITHM, algorithm_settings
import knn


def predict_thread(face_algorithm, model_name, image_file):
    # https://discuss.streamlit.io/t/attributeerror-thread-local-object-has-no-attribute-value/574/3
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    return knn.predict(image_file, 
        model_path=model_name, 
        distance_threshold=ALGORITHM[face_algorithm]['distance_threshold'],
        face_algorithm=face_algorithm)


def predict_parallel(image_file):
    all_predictions = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(predict_thread, algorithm_settings[1][0], algorithm_settings[1][1], image_file)
        future2 = executor.submit(predict_thread, algorithm_settings[2][0], algorithm_settings[2][1], image_file)
        for future in concurrent.futures.as_completed([future1, future2]):
            predictions = future.result()
            if future==future1:
                all_predictions[1] = predictions
            else:
                all_predictions[2] = predictions
    
    #print(all_predictions)

    # 综合结果判断：
    # 1. 如果两个结果唯一且相同，则无异议
    # 2. 如果都为unkonw，则无结果
    # 3. 如果有一个为unknown， 则返回非unknown的
    # 4. 如果有一个为multi, 则返回非multi的
    # 5. 如果都是multi, 优先返回算法1的
    # 6. 如果两个都有唯一结果，优先返回算法1的
    # 7. 如果结果都为0，则无结果
    # 8. 如果有一个为0， 则返回非0的
    # 9. 最后优先返回算法1的结果
    final_result=[]
    len1 = len(all_predictions[1])
    len2 = len(all_predictions[2])
    name1=name2=''
    if len1>0:
        name1 = all_predictions[1][0][0]
    if len2>0:
        name2 = all_predictions[2][0][0]

    # 条件 7
    if len1==len2==0:
        final_result=[]
    # 条件 8
    elif 0 in (len1, len2):
        if len2==0:
            final_result = all_predictions[1]
        else:
            final_result = all_predictions[2]
    # 条件 2
    elif name1==name2=='unknown': 
        final_result = all_predictions[1]
    # 条件 3
    elif 'unknown' in (name1, name2): 
        if name1=='unknown':
            final_result = all_predictions[2]
        else:
            final_result = all_predictions[1]
    # 条件 1, 6
    elif len1==len2==1: 
        final_result = all_predictions[1]
    # 条件 4, 5
    elif len1>1 or len2>1:
        if len2==1:
            final_result = all_predictions[2]
        else:
            final_result = all_predictions[1]
    # 条件 9
    else:
        final_result = all_predictions[1]
    return final_result



if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python3 %s <test dir or file>" % sys.argv[0])
        sys.exit(2)

    test_thing = sys.argv[1]

    if os.path.isdir(test_thing):
        images = os.listdir(test_thing)
        images = [os.path.join(test_thing, i) for i in images]
    else:
        images = [ test_thing ]

    # Using the trained classifier, make predictions for unknown images
    for image_file in images:
        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        
        start_time = datetime.now()
        predictions = predict_parallel(image_file)
        print('[Time taken: {!s}]'.format(datetime.now() - start_time))

        # Print results on the console
        for name, (top, right, bottom, left), distance, count in predictions:
            print("- Found {} at ({}, {}), distance={}, count={}".format(name, left, top, distance, count))
        if len(predictions)==0:
            print('Face not found!')

        # Display results overlaid on an image
        #knn.show_prediction_labels_on_image(image_file, predictions)


