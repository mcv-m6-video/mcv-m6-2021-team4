import cv2
import numpy as np
from tqdm import tqdm
import sys


from aicity_reader import read_annotations, read_detections, group_by_frame


def filter_bboxes_size(det_bboxes):
    filtered_bboxes = []
    for b in det_bboxes:
        if b.width >= 70 and b.height >= 56:
            filtered_bboxes.append(b)

    return filtered_bboxes

colors = {
    'r': (0, 0, 255),
    'g': (0, 255, 0),
    'b': (255, 0, 0),
    'w': (255,255,255)
    }
color_ids = {'16': (0, 0, 0),
    '63': (0, 0, 255),
    '69': (0, 255, 0),
    '202': (255,0,0),
    '482': (255, 255, 255),
    '746': (255, 255, 0),
    '903': (255, 0, 255),
    '1044': (0,255,255),
    }


def draw_boxes(image, boxes, tracker,  color='g', linewidth=5, det=False, boxIds=False, old=False):
    rgb = colors[color]
    for box in boxes:
        ##print(box.id)
        if boxIds:
            if box.id in list(color_ids.keys()):
                pass
            else:
                color_ids[box.id]=np.random.uniform(0,256,size=3)
            if old:
                cv2.putText(image, str(box.id), (int(box.xtl), int(box.ytl) + 120), cv2.FONT_ITALIC, 0.6, color_ids[box.id], linewidth)
            else:
                cv2.putText(image, str(box.id), (int(box.xtl), int(box.ytl) + 20), cv2.FONT_ITALIC, 0.6, color_ids[box.id], linewidth)

            #if len(tracker[box.id])>2:
             #   image =cv2.polylines(image,[np.array(tracker[box.id])],False,color_ids[box.id],linewidth)

            # if len(kalman_predictions[box.id])>2:
            #     image =cv2.polylines(image,[np.array(kalman_predictions[box.id])],False,color_ids[box.id],linewidth)


            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), color_ids[box.id], linewidth)
        else:
            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), rgb, linewidth)

        if det:
            cv2.putText(image, str(box.confidence), (int(box.xtl), int(box.ytl) - 5), cv2.FONT_ITALIC, 0.6, rgb, linewidth)
       
    return image
if __name__ == "__main__":
    params = {
        'data_path': "D:\\Ian\\UNI\\5_Master_CV\\M6\\Project\\Data\\aicity2021\\AIC21_Track3_MTMC_Tracking\\train\\S03\\",
        # 'det_path': "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt", #MASK RCNN
        #'roi_path': "./data/AICity_data/train/S03/c010/roi.jpg",

    }


    cams = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    
    # All the videos are 10 FPS, except c015, that is 8 FPS
    fps_ratio = {
        'c010': 1.0,
        'c011': 1.0,
        'c012': 1.0,
        'c013': 1.0,
        'c014': 1.0,
        'c015': 10.0 / 8.0
    }
    frame_offset = {
        'c010': 87, # 8.715s * 10 fps 
        'c011': 84, # 8.457s * 10 fps 
        'c012': 59, # 5.879s * 10 fps 
        'c013': 0,
        'c014': 50, # 5.042s * 10 fps 
        'c015': 68 # 8.492s * 8 fps 
    }

    cam_dicts = {}
    
    for cam in cams:
        vidcap = cv2.VideoCapture(params['data_path']+"{}\\vdo.avi".format(cam))
        #vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset[cam])           
        
        det = read_detections(params['data_path']+"{}\\overlap_reid_detections.txt".format(cam), grouped=True, confidenceThr=0.4)
        
        cam_dicts[cam] = [vidcap,int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), det]
        
        print("Cam: {}, Total frames: {}, fps: {}, frame_offset: {}".format(cam, cam_dicts[cam][1], 10/fps_ratio[cam],frame_offset[cam]))
        
    frame_id = 0   
    frames = {}
    last_frame = np.zeros(shape=(1080,1920,3))
    ids = []
    while(True):
        if frame_id < 400:
            for cam in cams:
                if frame_id > + frame_offset[cam] and frame_id < cam_dicts[cam][1]+ frame_offset[cam]:
                    _, frame = cam_dicts[cam][0].read()
                    
                    det_bboxes = []
                    if (frame_id-frame_offset[cam]) in cam_dicts[cam][2]:
                        det_bboxes = filter_bboxes_size(cam_dicts[cam][2][frame_id-frame_offset[cam]])
                        #for d in det_bboxes:
                            #print("cam: {} frame: {}, frame_ids: {} ".format(cam, frame_id,d.id))
                            #ids.append(d.id)
                    
                    
                    ##frame = draw_boxes_old(image=frame, boxes=det_bboxes, tracker=None,  color='g', linewidth=2)
                    frame = draw_boxes(image=frame, boxes=det_bboxes, tracker=None,boxIds = True,  color='g', linewidth=5)
                    frame = cv2.resize(frame,(1920,1080))
                    frames[cam] = frame
                    ##last_frame = frame
                else:
                    frames[cam] = last_frame
                    

            # concatanate image Horizontally
            hor1 = np.concatenate((frames['c013'], frames['c012'], frames['c011']), axis=1)
            hor2 = np.concatenate((frames['c014'], frames['c015'], frames['c010']), axis=1)  
            ## concatanate image Vertically
            vert = np.concatenate((hor2, hor1), axis=0)
            vert = cv2.resize(vert,(1920,1080))
            
            '''
            cv2.imshow('VERTICAL', vert)
            cv2.waitKey(10)
            '''
          
            cv2.imwrite('D:\\Ian\\UNI\\5_Master_CV\\M6\\Project\\week_5\\results\\allcams_track\\{}.png'.format(10000+frame_id),vert)
            
            
            frame_id += 1
            if frame_id >= cam_dicts['c012'][1]+ frame_offset['c012']:
                break
        else:
            break
