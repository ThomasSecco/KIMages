from ultralytics import YOLO
import torch


def yolo(img):
    model=YOLO('yolov8x-cls.pt')
    result=model(img)
    res=[]
    names=result[0].names
    probs=result[0].probs
    for i in range (5):
        if probs.top5conf[i]>0.5:
            res.append((names[probs.top5[i]],probs.top5conf[i]))
    if res==[]:
        res.append((names[probs.top1],probs.top1conf))
    return res

