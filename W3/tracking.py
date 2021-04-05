import sys
sys.path.append("W1")
from voc_evaluation import voc_iou_tracking

class Tracking:

    def __init__(self):
        self.last_id_assigned = -1

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
                setattr(bb, 'id', self.new_id()) #Set BB class id
                #TODO
            
            return det_bboxes_new
        else:
            # if len(det_bboxes_new)> len(det_bboxes_prev):
            #     print("More new than old")
            # elif len(det_bboxes_new)< len(det_bboxes_prev):
            #     print("more old than new")
            # else:
            #     print("equal det")
            
            matches = {}
            for i,bb in enumerate(det_bboxes_new):
                max_iou = 0
                matched_bb_id = -1
                match_found = False
                
                for bb_prev in det_bboxes_prev:

                    if bb_prev.flow is not None:
                        # print("Using Optical Flow")
                        # bb_prev.apply_flow()
                        # print("")
                        # print(bb_prev.box)
                        # print(bb_prev.box_flow)
                        iou = voc_iou_tracking(bb.box, bb_prev.box_flow)
                    else:
                        # print("Not using Optical Flow")
                        iou = voc_iou_tracking(bb.box, bb_prev.box)


                    if iou > max_iou:
                        match_found = True
                        max_iou = iou
                        # matches[bb] = {bb_prev.id: max_iou}
                        if bb_prev.id in matches.keys():
                            matches[bb_prev.id] += [[i,max_iou]]
                        else:
                            matches[bb_prev.id] = [[i,max_iou]]
                if not match_found:
                    setattr(det_bboxes_new[i], 'id', self.new_id())
                        
                        
        
            id_assigned = []

            for items in list(matches.items()):
                if len(items[1])>= 2:
                    sorted_list= sorted(items[1],key=lambda x: x[1],reverse=True)
                    bb_to_assign = sorted_list[0][0]
                    id_assigned.append(bb_to_assign)
                    setattr(det_bboxes_new[bb_to_assign], 'id', items[0])

                else:
                    bb_to_assign = items[1][0][0]
                    id_assigned.append(bb_to_assign)
                    setattr(det_bboxes_new[bb_to_assign], 'id', items[0])

            for n in range(len(det_bboxes_new)):
                if n not in id_assigned:
                    setattr(det_bboxes_new[n], 'id', self.new_id())


                #     
                

            # else: 
            #     for items in list(matches.items()):
            #     #TODO
            #     return det_bboxes_prev

            # print(matches)
            # print("ei")
                    # print(iou)
        return det_bboxes_new

