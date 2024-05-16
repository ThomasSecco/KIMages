
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch
import numpy as np
import cv2
import os
import shutil

# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Class labels from official PyTorch documentation for the pretrained model
# Note that there are some N/A's 
# for complete list check https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# we will use the same list for this notebook
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

def get_prediction(img_path, threshold):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
    
  """
  import torchvision.transforms as T
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  pred_score = list(pred[0]['scores'].detach().numpy())
  filtered_scores = [x for x in pred_score if x > threshold]
  if filtered_scores:
      pred_t = pred_score.index(filtered_scores[-1])
  else:
      pred_t = -1  # or any default value indicating no predictions meet the threshold
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class
  


def object_detection_and_save(img_path, save_dir, threshold=0.5):
    """
    object_detection_and_save
      parameters:
        - img_path: path of the input image
        - save_dir: directory to save the cropped bounding box images
        - threshold: threshold value for prediction score
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is used to crop the image
        - cropped images are saved in the specified directory
    """
    boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)  
    [os.remove(os.path.join(save_dir, f)) for f in os.listdir(save_dir)]      

    for i, box in enumerate(boxes):
        # Extract coordinates of bounding box
        (startX, startY) = (int(box[0][0]), int(box[0][1]))
        (endX, endY) = (int(box[1][0]), int(box[1][1]))
        
        # Crop the image using bounding box coordinates
        cropped_img = img[startY:endY, startX:endX]
        
        # Generate a filename for the cropped image
        filename = os.path.join(save_dir, f"cropped_image_{i}.jpg")
        
        # Save the cropped image
        cv2.imwrite(filename, cropped_img)

