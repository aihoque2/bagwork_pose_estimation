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

from ai_code.model import dope_resnet50, num_joints

import torch
from torchvision.transforms import ToTensor

import cv2

_thisdir = osp.realpath(osp.dirname(__file__)) #TODO: wtf this do?


def load_model(modelname):
    """
    load which model to use for psoe estimation given the name
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpi'

    cpkft_fname = osp.join("/ai_code/models/" + modelname)
    
    #error checking case
    if not os.path.isfile(ckpt_fname):
        raise Exception('{:s} does not exist, please download the model first and place it in the models/ folder'.format(ckpt_fname))
    
    ckpt = torch.load(cpkft_fname, map_location=device)
    ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
    model = dope_resnet50(**ckpt['dope_kwargs'])
    if ckpt['half']: model = model.half()
    model = model.eval()
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)

    return model #remember objects get returned by reference

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