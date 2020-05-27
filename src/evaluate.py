# -*- coding: utf-8 -*-

import os, sys
from datetime import datetime
from settings import ALGORITHM
import knn


if __name__ == "__main__":
    if len(sys.argv)<4:
        print("usage: python3 %s <algorithm> <model_name> <test dir>" % sys.argv[0])
        sys.exit(2)

    face_algorithm = sys.argv[1]

    if face_algorithm not in ALGORITHM.keys():
        print('Algorithm not found!')
        sys.exit(2)

    test_path = sys.argv[3]
    model_name = sys.argv[2]

    if not model_name.endswith(ALGORITHM[face_algorithm]['ext']):
        model_name += ALGORITHM[face_algorithm]['ext']

    if not os.path.isdir(test_path):
        print('test need directory.')
        sys.exit(0)

    persons = os.listdir(test_path)
    total_acc = total_acc2 = 0

    # name      total   correct wrong   fail    multi     second        acc      acc2            preci        elapsed time
    # 样本名     总数     正确数   错误数   失败数   返回多结果  非第一结果正确   正确率    非第一结果正确率   非失败正确率    耗时
    print('name\t\ttotal\tcorrect\twrong\tfail\tmulti\tsecond\tacc\tacc2\tpreci\telapsed time')
    for p in persons:
        images = os.listdir(os.path.join(test_path, p))
        images = [os.path.join(test_path, p, i) for i in images]

        # Using the trained classifier, make predictions for unknown images
        total = len(images)
        correct = 0
        wrong = 0
        fail = 0
        multi = 0 # 匹配多个结果
        second = 0 # 不是首个匹配结果
        start_time = datetime.now()
        for image_file in images:
            #print("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = knn.predict(image_file, 
                model_path=model_name, 
                distance_threshold=ALGORITHM[face_algorithm]['distance_threshold'],
                face_algorithm=face_algorithm)

            # Print results on the console
            if len(predictions)==0:
                fail += 1
            else:
                n = 0
                bingo = 0
                name_list = []
                for name, (top, right, bottom, left), distance, count in predictions:
                    if name==p:
                        if n==0:
                            correct += 1
                            bingo += 1
                        elif bingo==0:
                            second += 1
                    else:
                        if n==0:
                            wrong += 1

                    if name not in name_list:
                        name_list.append(name)

                    n += 1

                if len(name_list)>1:
                    multi += 1


        print('%10s\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s'%\
            (p, total, correct, wrong, fail, multi, second, correct/total, (correct+second)/total, correct/(total-fail), datetime.now() - start_time))

        total_acc += correct/total
        total_acc2 += (correct+second)/total

    print('total_acc: %.3f'%(total_acc/len(persons)))
    print('total_acc2: %.3f'%(total_acc2/len(persons)))
