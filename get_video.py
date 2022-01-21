#script to get the video file
import cv2
import torch

def get_image_from_video(filename, index):
    """
    read the video with cv2 then select
    the frame the the particular index
    
    return the mat that represents the frame
    """
    cap = cv2.VideoCapture(filename)
    i = 0:
    while (i < index):
        i+=1
        frame = cap.read()
    return frame

def get_video(filename):
    pass