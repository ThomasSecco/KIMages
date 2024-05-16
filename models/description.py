import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from vect import simi
from models import captionning,classification,yolo,detection

def results(image):
    caption=captionning.show_n_generate(image)
    class1=classification.eval(image)
    class1=[tup[0] for tup in class1]
    class1=[x.replace('_',' ') for x in class1]
    class2=detection.get_prediction(image,0.5)[-1]
    class2=[x.replace('_',' ') for x in class2]
    class3=yolo.yolo(image)
    class3=[tup[0] for tup in class3]
    class3=[x.replace('_',' ') for x in class3]
    return caption,class1,class2,class3

def compare_lists(list1, list2, list3):
    found_in_all = []
    found_in_two = []
    found_in_one = []

    for class1 in list1:
        found = False
        for class2 in list2:
            for class3 in list3:
                sim_score_1_2 = simi(class1, class2)
                sim_score_1_3 = simi(class1, class3)
                sim_score_2_3 = simi(class2, class3)

                if sim_score_1_2 > 0.8 and sim_score_1_3 > 0.8 and sim_score_2_3 > 0.8:
                    found_in_all.append(class1)
                    found = True
                    break
                elif (sim_score_1_2 > 0.8 and sim_score_1_3 > 0.8) or (sim_score_1_2 > 0.8 and sim_score_2_3 > 0.8) or (sim_score_1_3 > 0.8 and sim_score_2_3 > 0.8):
                    found_in_two.append(class1)
                    found = True
                    break

            if found:
                break

        if not found:
            found_in_one.append(class1)

    # Add classes from list2 and list3 not already categorized
    for class2 in list2:
        if class2 not in found_in_all and class2 not in found_in_two:
            found_in_one.append(class2)

    for class3 in list3:
        if class3 not in found_in_all and class3 not in found_in_two:
            found_in_one.append(class3)

    return found_in_all, found_in_two, found_in_one

def score(found_in_all, found_in_two, found_in_one, caption):
    cap=caption.split()
    p1,p2,p3=0,0,0
    if found_in_all:
        for class_name in found_in_all:
            max_sim = 0
            for word in cap:
                sim = simi(class_name, word)
                if sim > max_sim:
                    max_sim = sim
            p1 += max_sim/len(found_in_all)

    if found_in_two:
        for class_name in found_in_two:
            max_sim = 0
            for word in cap:
                sim = simi(class_name, word)
                if sim > max_sim:
                    max_sim = sim
            p2 += max_sim/len(found_in_two)

    if found_in_one:
        for class_name in found_in_one:
            max_sim = 0
            for word in cap:
                sim = simi(class_name, word)
                if sim > max_sim:
                    max_sim = sim
            p3 += max_sim/len(found_in_one)
    
    if p1==0:
        if p2==0:
            total_score=p3
        elif p3==0:
            total_score=p2
        else:
            total_score=p2*0.7+p3*0.3
    elif p2==0:
        if p1==0:
            total_score=p3
        elif p3==0:
            total_score=p1
        else:
            total_score=p1*0.8+p3*0.2
    elif p3==0:
        if p1==0:
            total_score=p2
        elif p2==0:
            total_score=p1
        else:
            total_score=p1*0.6+p2*0.4
    else:
        total_score=p1*0.6+p2*0.3+p3*0.1

    
    return caption,100*total_score

def details(image,res):#res is the output of the results function above
    a=[]
    detection.object_detection_and_save(image,'c:/Stage/Furtwangen/models/im')
    for i in range (len(res[2])):
        for x in res[0].split():
            if simi(res[2][i],x)>0.5:
                a.append(captionning.show_n_generate(f'c:/Stage/Furtwangen/models/im/cropped_image_{i}.jpg'))
    return np.unique(a).tolist()