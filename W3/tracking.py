import sys
sys.path.append("W1")
from voc_evaluation import voc_iou_tracking

class Tracking:

    def __init__(self):
        self.last_id_assigned = 0

    @property
    def get_last_id(self):
        return self.last_id_assigned

    def new_id(self):
        self.last_id_assigned += 1
        return self.last_id_assigned
        
    
    def set_frame_ids(self, det_bboxes_new, det_bboxes_prev):

        if det_bboxes_prev == -1: 
            for i, bb in enumerate(det_bboxes_new):
                # print(i)
                setattr(bb, 'id', i) #Set BB class id
                #TODO
            return
        else:
            
            matches = {}
            for bb in det_bboxes_new:
                max_iou = 0
                matched_bb_id = -1
                
                for bb_prev in det_bboxes_prev:
                    iou = voc_iou_tracking(bb.box, bb_prev.box)

                    if iou > max_iou:
                        max_iou = iou
                        matches[bb] = {bb_prev.id: max_iou}

            if len(det_bboxes_new) > len(det_bboxes_prev):
                setattr(bb, 'id', bb_prev.id)
                #TODO
                pass

            else: 
                #TODO
                pass

            print(matches)
            print("ei")
                    # print(iou)

