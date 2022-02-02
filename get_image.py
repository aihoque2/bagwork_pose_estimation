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

    ##NOTE: CAPTURING VIDEO FRAMES:
    ##in order to capture a video frame,

    i = 0
    success = 1
    while (success and i <= index):
        success, frame = cap.read()
        if (i==index):
            return frame
        i+=1

def get_video(filename):
    pass