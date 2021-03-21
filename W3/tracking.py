img_shape = [1080, 1920]


class Tracking:

    def __init__(self):
        self.last_id_assigned = 0

    @property
    def get_last_id(self):
        return self.last_id_assigned

    def new_id(self):
        self.last_id_assigned += 1
        return self.last_id_assigned
        
    
    def set_frame_ids(self, det_bboxes):

        for bb in det_bboxes:
            print(bb)
            setattr(bb, 'id', 2) #Set BB class id
            #TODO
        return
