# Importing the YOLO class from ultralytics and torch module
from ultralytics import YOLO
import torch

def yolo(img):
    """
    Perform object detection on an image using a pre-trained YOLO model.
    
    The function takes an image as input, processes it through a YOLO model, and returns a list of detected objects
    with high confidence scores. It focuses on providing the most likely objects detected in the image with a confidence
    threshold of 50%. If no objects meet this threshold, the top predicted object is returned regardless of its confidence.

    Parameters:
        img : A string showing the path to the image file.
    
    Returns:
        list of tuples: Each tuple contains a detected object's name and its confidence level. The list includes only
                        high-confidence detections or the highest confidence detection if no detections are above the threshold.
    """

    # Load the YOLO model with pre-trained weights from 'yolov8x-cls.pt'
    model = YOLO('yolov8x-cls.pt')

    # Perform object detection on the input image
    result = model(img)

    # Initialize an empty list to store results
    res = []

    # Accessing class names and probability scores from the detection results
    names = result[0].names
    probs = result[0].probs

    # Iterate over the top 5 detected objects based on confidence
    for i in range(5):
        # Check if the confidence level of the detected object is greater than 50%
        if probs.top5conf[i] > 0.5:
            # If yes, append the object name and its confidence to the results list
            res.append((names[probs.top5[i]], probs.top5conf[i]))

    # If no objects have a confidence greater than 50%, add the top 1 object
    if res == []:
        res.append((names[probs.top1], probs.top1conf))

    # Return the final list of detected objects with their names and confidence levels
    return res
