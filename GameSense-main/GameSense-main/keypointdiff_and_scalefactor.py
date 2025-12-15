# from .court_reference import CourtReference
# from .court_detection_net import CourtDetectorNet
import numpy as np
try:
    import torch
except Exception:
    torch = None

PIXEL_TO_METER = 0.0002645833

# def calculate_pixel_difference(kp1, kp2):
#     """
#     Calculate the Euclidean distance between two keypoints in pixels.
#     :param kp1: Tuple (x1, y1) - coordinates of the first keypoint.
#     :param kp2: Tuple (x2, y2) - coordinates of the second keypoint.
#     :return: Euclidean distance in pixels.
#     """
#     return np.sqrt((kp2[0] - kp1[0])**2 + (kp2[1] - kp1[1])**2)

# def convert_pixel_to_metre():
#     return pixel_dist*PIXEL_TO_METER

def keypoint_differences_in_metres(kps_court):
    if kps_court is None or kps_court[0] is None:
        return 1.0, 1.0, 1.0, 1.0
    i = 0
    x1 = kps_court[i][0][0,0]
    y1 = kps_court[i][0][0,1]
    x2 = kps_court[i][1][0,0]
    y2 = kps_court[i][1][0,1]
    x3 = kps_court[i][2][0,0]
    y3 = kps_court[i][2][0,1]
    x4 = kps_court[i][3][0,0]
    y4 = kps_court[i][3][0,1]
    x5 = kps_court[i][6][0,0]
    y5 = kps_court[i][6][0,1]
    x6 = kps_court[i][7][0,0]
    y6 = kps_court[i][7][0,1]
    

    
    
    dx1 = (x2-x1)**2
    dx2 = (x3-x4)**2
    dy1 = (y2-y1)**2
    dy2 = (y3-y4)**2
    dx3 = (x5-x6)**2
    dy3 = (y5-y6)**2
    pixdist1 = (dx1+dy1)**0.5 #Away
    pixdist2 = (dx2+dy2)**0.5 #Near
    pixdist3 = (dx3+dy3)**0.5 #for Ball  
    metdist1 = pixdist1*PIXEL_TO_METER
    metdist2 = pixdist2*PIXEL_TO_METER
    metdist3 = pixdist3*PIXEL_TO_METER
    return metdist1, metdist2, metdist3, pixdist1/pixdist2

def scaleFactor(kps_court):
    if kps_court is None:
        return 1.0, 1.0, 1.0
    metdist1, metdist2, metdist3, ratio = keypoint_differences_in_metres(kps_court)
    # Avoid division by zero
    if metdist1 == 0 or metdist2 == 0 or metdist3 == 0:
        return 1.0, 1.0, 1.0
    scale1 = 10.97/metdist1
    scale2 = 10.97/metdist2
    scale3 = 23.77/metdist3
    return scale1, scale2, scale3
    

def centre_court(kps_court):
    if kps_court is None or kps_court[0] is None:
        return (0, 0)
    i = 0
    x1 = kps_court[i][0][0,0]
    y1 = kps_court[i][0][0,1]
    x2 = kps_court[i][1][0,0]
    y2 = kps_court[i][1][0,1]
    x3 = kps_court[i][2][0,0]
    y3 = kps_court[i][2][0,1]
    x4 = kps_court[i][3][0,0]
    y4 = kps_court[i][3][0,1]
    
    x_centre = x1+x2+x3+x4/4
    y_centre = y1+y2+y3+y4
    
    return (x_centre, y_centre)

