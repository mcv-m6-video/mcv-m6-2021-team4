import numpy as np
import cv2
import sys
sys.path.append(".//W1")
from bounding_box import BoundingBox


color_space = {
    'grayscale':[cv2.COLOR_BGR2GRAY,1],
    'RGB': [cv2.COLOR_BGR2RGB,3],
    'HSV': [cv2.COLOR_BGR2HSV,3],
    'LAB': [cv2.COLOR_BGR2LAB,3],
    'YUV': [cv2.COLOR_BGR2YUV,3],
    'YCrCb': [cv2.COLOR_BGR2YCrCb,3]
}

'''

def static_bg_est_old(image, mean, std, params):
    alpha = params['alpha']
    segmentation = np.zeros((1080, 1920))
    segmentation[abs(image - mean) >= alpha * (std + 2)] = 255
    return segmentation, mean, std


def adaptive_bg_est_old(image,frame_size, mean, std, params):
    alpha = params['alpha']
    rho = params['rho']

    mask = abs(image - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    segmentation[mask] = 255

    # provar d'actualitzar el bg cada X frames
    mean = np.where(mask, mean, rho * image + (1-rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (image-mean)**2 + (1-rho) * std**2))

    return segmentation, mean, std


'''
def static_bg_est(image,frame_size, mean, std, params):
    alpha = params['alpha']
    h,w = frame_size
    segmentation = np.zeros((h, w))
    mask = abs(image - mean) >= alpha * (std + 2)
    
    nc = color_space[params['color_space']][1]
    if nc == 1:
        segmentation[mask] = 255
    else:
        if params['voting'] == 'unanimous' or nc==2: 
            voting = (np.count_nonzero(mask,axis=2)/nc)>=1   
        elif params['voting'] == 'simple':
            voting = (np.count_nonzero(mask,axis=2)/(nc//2 + 1))>=1
        else:
            raise ValueError('Voting method does not exist')
        
        segmentation[voting] = 255
        
    return segmentation, mean, std

def adaptive_bg_est(image,frame_size, mean, std, params):
    alpha = params['alpha']
    rho = params['rho']
    h,w = frame_size
    mask = abs(image - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    nc = color_space[params['color_space']][1]
    
    if nc == 1:
        segmentation[mask] = 255
    else:
        if params['voting'] == 'unanimous' or nc==2: 
            voting = (np.count_nonzero(mask,axis=2)/nc)>=1   
        elif params['voting'] == 'simple':
            voting = (np.count_nonzero(mask,axis=2)/(nc//2 + 1))>=1
        else:
            raise ValueError('Voting method does not exist')
        
        segmentation[voting] = 255
    
    # provar d'actualitzar el bg cada X frames
    mean = np.where(mask, mean, rho * image + (1-rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (image-mean)**2 + (1-rho) * std**2))

    return segmentation, mean, std


def postprocess_fg(seg):
    kernel = np.ones((2, 2), np.uint8)
    # seg = cv2.erode(seg, kernel, iterations=1)
    # seg = cv2.dilate(seg, kernel, iterations=1)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel)
    return seg

def fg_bboxes(seg, frame_id):
    bboxes = []
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50:
            continue  # Discard small contours

        x, y, w, h = rect

        # TODO: pillar bicis a gt
        bboxes.append(BoundingBox(id=idx, label='car', frame=frame_id, xtl=x,
                                  ytl=y, xbr=x+w, ybr=y+h))
        idx += 1

    return bboxes