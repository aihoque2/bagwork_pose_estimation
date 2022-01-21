#main.py
"""
pose track a video
of me boxing

use the DOPE estimator
as seen here:

https://arxiv.org/pdf/2008.09457.pdf
"""
import cv2
from get_video import get_image_from_video

filename = "video/jab.mp4"
mat = get_image_from_video(filename, 0)
cv2.imwrite("standing.jpg", mat)