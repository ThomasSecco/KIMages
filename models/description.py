import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from . import captionning, classification, yolo, detection,vect
from .vect import simi  # Custom module for similarity calculation

def results(image):
    """
    Aggregates classification and detection results from multiple models on a given image.

    Parameters:
        image (str): Path to the image file.

    Returns:
        tuple: Contains the caption of the image, and classification results from three different models.
    """
    # Generate caption for the image
    caption = captionning.show_n_generate(image)

    # Get class predictions from the classification model
    class1 = classification.eval(image)
    class1 = [tup[0].replace('_', ' ') for tup in class1]  # Clean up class names

    # Get class predictions from the detection model
    class2 = detection.get_prediction(image, 0.5)[-1]
    class2 = [x.replace('_', ' ') for x in class2]  # Clean up class names

    # Get class predictions from the YOLO model
    class3 = yolo.yolo(image)
    class3 = [tup[0].replace('_', ' ') for tup in class3]  # Clean up class names

    return caption, class1, class2, class3

def compare_lists(list1, list2, list3):
    """
    Compares three lists of classes and categorizes them based on the similarity of their contents.

    Parameters:
        list1, list2, list3 (list): Lists of class names from different models.

    Returns:
        tuple: Contains lists of class names found in all three lists, two lists, and one list.
    """
    found_in_all, found_in_two, found_in_one = [], [], []

    # Compare each class in list1 with classes in list2 and list3
    for class1 in list1:
        found = False
        for class2 in list2:
            for class3 in list3:
                sim_score_1_2 = simi(class1, class2)
                sim_score_1_3 = simi(class1, class3)
                sim_score_2_3 = simi(class2, class3)

                # Check similarity threshold to decide in how many lists the class appears
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

    # Check for unmatched classes in list2 and list3
    for class2 in list2:
        if class2 not in found_in_all and class2 not in found_in_two:
            found_in_one.append(class2)
    for class3 in list3:
        if class3 not in found_in_all and class3 not in found_in_two:
            found_in_one.append(class3)

    return found_in_all, found_in_two, found_in_one

def score(found_in_all, found_in_two, found_in_one, caption):
    """
    Computes a weighted score based on how many lists the classes appear in and their similarity to the caption.

    Parameters:
        found_in_all, found_in_two, found_in_one (list): Lists of class names sorted by presence in 3, 2, or 1 list.
        caption (str): Caption of the image.

    Returns:
        tuple: The caption and the computed score as a percentage.
    """
    cap = caption.split()
    p1, p2, p3 = 0, 0, 0

    # Calculate partial scores based on similarity with words in the caption
    if found_in_all:
        p1 = sum(max(simi(class_name, word) for word in cap) for class_name in found_in_all) / len(found_in_all)
    if found_in_two:
        p2 = sum(max(simi(class_name, word) for word in cap) for class_name in found_in_two) / len(found_in_two)
    if found_in_one:
        p3 = sum(max(simi(class_name, word) for word in cap) for class_name in found_in_one) / len(found_in_one)
    
    # Compute total weighted score based on distribution across lists
    total_score = p1 * 0.6 + p2 * 0.3 + p3 * 0.1

    return caption, 100 * total_score
