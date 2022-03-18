"""
mediapipe_test.py

as a software developer, we must always try new
libraries. This is where I try mediapipe for pose
estimnation.
"""

import cv2
import medapipe as mp

###global mediapipe bariables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def run_blazepose(img):
    """
    run the BalzePose SOTA model
    (Or is it state of the art?)

    https://arxiv.org/pdf/2006.10204.pdf
    """
    pass
