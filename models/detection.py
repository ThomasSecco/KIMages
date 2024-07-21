from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2
import os


# Load a pre-trained Faster R-CNN model with a ResNet50 backbone and FPN from torchvision.
# Set the model in evaluation mode for inference.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# List of class labels used in the COCO dataset for object detection.
# Includes some placeholders for non-applicable categories ('N/A').
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_prediction(img_path: str, threshold: float):
    """
    Get predictions for an image based on a specified threshold for detection confidence.

    Parameters:
        img_path (str): Path to the input image.
        threshold (float): Minimum confidence score for predictions to be considered.

    Returns:
        Tuple of lists: A list of bounding box coordinates and a list of corresponding class labels.
    """
    import torchvision.transforms as T
    # Open the image file, convert it to a tensor using PyTorch transforms.
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    
    # Make predictions using the pre-trained model.
    pred = model([img])
    
    # Extract class labels, bounding box coordinates, and confidence scores.
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    
    # Filter predictions based on the threshold.
    filtered_scores = [x for x in pred_score if x > threshold]
    pred_t = pred_score.index(filtered_scores[-1]) if filtered_scores else -1
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    return pred_boxes, pred_class


