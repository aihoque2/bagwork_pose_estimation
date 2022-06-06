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

import ai_code.postprocess as postprocess
import ai_code.visu as visu
from ai_code.model import Dope_RCNN, dope_resnet50, num_joints


import torch
from torchvision.transforms import ToTensor

import cv2

_thisdir = osp.realpath(osp.dirname(__file__)) #TODO: wtf this do?

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#QUESTION FOR CODERS: should  implement each function and test
#or should i implement all at once?

def load_model(modelname, postprocessing='ppi'):
    """
    load which model to use for psoe estimation given the name
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpi'

    cpkft_fname = osp.join("/ai_code/models/" + modelname)
    
    #error checking case
    if not os.path.isfile(ckpt_fname):
        raise Exception('{:s} does not exist, please download the model first and place it in the models/ folder'.format(ckpt_fname))
    
    ckpt_fname = osp.join(_thisdir, 'models', modelname+'.pth.tgz')
    ckpt = torch.load(ckpt_fname, map_location=device)
    print("WE GOT ckpt: ", ckpt)

    ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
    model = dope_resnet50(**ckpt['dope_kwargs'])

    #TODO: Check if we support half-computation
    if ckpt['half']: 
        model = model.half()
    model = model.eval()
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    return model

def run_dope_on_mat_NMS(image: Image, model: Dope_RCNN, postprocessing="ppi"):
    """
    TODO: run a simple pose estimation with the given model.
    this function is like dope() in dope.py, but we add another
    dimension 

    this assumes model has been loaded
    """
    ckpt_fname = osp.join(_thisdir, 'models', modelname+'.pth.tgz')
    ckpt = torch.load(ckpt_fname, map_location=device)

    img_tensor = [ToTensor()(image).to(device)]
    if ckpt['half']: imlist = [im.half() for im in imlist]
    resolution = imlist[0].size()[-2:]

    #run DOPE
    with torch.no_grad():
        results = model(imlist, None)[0]
    
    #TODO: we go the results now how do we extract them?
    #NMS method...2D pose
    
    parts = ["body", "hand", "face"]
    detections = {}
    
    for part in parts:
        dets, indices, bestcls = postprocess.DOPE_NMS(results[part+'_scores'], results['boxes'], results[part+'_pose2d'], results[part+'_pose3d'], min_score=0.3)
        dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
        detections[part] = [{'score': dets['score'][i], 'pose2d': dets['pose2d'][i,...], 'pose3d': dets['pose3d'][i,...]} for i in range(dets['score'].size)]
        if part == 'hand':
            for i in range(len(detections[part])):
                detections[part][i]['hand_isright'] = bestcls<ckpt['hand_ppi_kwargs']['K']

    # assignment of hands and head to body
    detections, body_with_wrists, body_with_head = postprocess.assign_hands_and_head_to_body(detections)

    #creating mat to return
    det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections > 0) else np.empty( (0, num_joints[part], 2), dtype=np.float32) for part, part_detections in detections.items()}
    scores = {part: [d['score'] for d in part_detections] for part, part_detections in detections.items()}
    imout = visu.visualize_bodyhandface2d(np.asarray(image)[:,:,::-1],
                                            det_poses2d,
                                            dict_scores=scores,
                                            )
    return imout

def process_video(filename, video):
    """
    TODO: take video file and convert in to CV mat.
    after converting CV mat, run mat through dope predictor
    fucntion for running dope can be found in dope.py
    """
    pass