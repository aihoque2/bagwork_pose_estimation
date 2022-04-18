"""
video_processor.py
video processing functions to use to create the new video
"""
import sys, os
import argparse
import os.path as osp
from PIL import Image
import cv2
import numpy as np

import torch
from torchvision.transforms import ToTensor

import cv2

def load_model(modelname):
    """
    load which model to use for psoe estimation given the name
    """
    pass

def run_dope_on_mat(image, model, postprocessing="ppi"):
    """
    TODO: run a simple pose estimation with the given model.
    this function is like dope() in dope.py, but we add another
    dimension 

    this assumes model has been loaded
    """
    pass



def process_video(filename, video):
    """
    TODO: take video file and convert in to CV mat.
    after converting CV mat, run mat through dope predictor
    fucntion for running dope can be found in dope.py
    """
    pass