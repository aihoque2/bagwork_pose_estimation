#script to get the frame from file
import cv2
import torch

def get_image_from_video(filename, index):
    """
    read the video with cv2 then select
    the frame the the particular index
    
    return the mat that represents the frame
    """
    cap = cv2.VideoCapture(filename)
    i = 0
    while (i <= index):
        frame = cap.read()
        if (i==index):
            return frame
        i+=1

def get_video(filename):
    pass