img_shape = [1080, 1920]


class BoundingBox:

    def __init__(self, id, label, frame, xtl, ytl, xbr, ybr, occluded=None, parked=None, confidence=None):
        self.id = id
        self.label = label
        self.frame = frame
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.occluded = occluded
        self.parked = parked
        self.confidence = confidence
        self.flow = None
        # self.center_pos = self.center() # to compute parked cars

    @property
    def box(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]
    
    @property
    def box_flow(self):
        # print(self.flow[0], self.flow[1])
        return [
            self.xtl + self.flow[0], 
            self.ytl + self.flow[1], 
            self.xbr + self.flow[0], 
            self.ybr + self.flow[1]]

    @property
    def center(self):
        return [(self.xtl + self.xbr) // 2, (self.ytl + self.ybr) // 2]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ybr - self.ytl)

    @property
    def area(self):
        return self.width * self.height

    def shift_position(self, center):
        w = self.width
        h = self.height
        self.xtl = center[0] - w/2
        self.ytl = center[1] - h/2
        self.xbr = center[0] + w/2
        self.ybr = center[1] + h/2
        return

    def resize(self, size):
        h, w = size
        c = self.center
        self.xtl = c[0] - w/2
        self.ytl = c[1] - h/2
        self.xbr = c[0] + w/2
        self.ybr = c[1] + h/2
        return

    def apply_flow(self):
        self.xtl += self.flow[0] 
        self.ytl += self.flow[1] 
        self.xbr += self.flow[0]
        self.ybr += self.flow[1]

    def point_inside_bbox(self, point):
        '''point = [x,y]'''
        if (point[0]>=self.xtl and point[0]<=self.xbr and point[1]>=self.ytl and point[1]<=self.ybr):
            return True
        else:
            return False
        
    # tmp name
    def inside_image(self):
        h, w = img_shape

        if self.xtl < 0:
            self.xtl = 0

        if self.ytl < 0:
            self.ytl = 0

        if self.xbr >= w:
            self.xbr = w - 1

        if self.ybr >= h:
            self.ybr = h - 1

        return


def intersection_bboxes(bboxA, bboxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bboxA.xtl, bboxB.xtl)
    yA = max(bboxA.ytl, bboxB.ytl)
    xB = min(bboxA.xbr, bboxB.xbr)
    yB = min(bboxA.ybr, bboxB.ybr)
    # return the area of intersection rectangle
    return max(0, xB - xA) * max(0, yB - yA)


def intersection_over_union(bboxA, bboxB):
    interArea = intersection_bboxes(bboxA, bboxB)
    iou = interArea / float(bboxA.area + bboxB.area - interArea)
    return iou


def intersection_over_areas(bboxA, bboxB):
    interArea = intersection_bboxes(bboxA, bboxB)
    return interArea / bboxA.area, interArea / bboxB.area