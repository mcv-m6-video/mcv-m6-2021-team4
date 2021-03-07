import xmltodict
import cv2

# import ffmpeg
# (
#     ffmpeg
#         .input('./data/AICity_data/train/S03/c010/vdo.avi')
#         .filter('fps', fps=1)
#         .output('test/%d.png',
#                 start_number=0)
#         .run(capture_stdout=True, capture_stderr=True)
#  )

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

    def box(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    def center(self):
        return [(self.xtl + self.xbr) // 2, (self.ytl + self.ybr) // 2]

    def width(self):
        return abs(self.xbr - self.xtl)

    def height(self):
        return abs(self.ybr - self.ytl)

    def area(self):
        return self.width * self.height

def read_annotations(path):
    with open(path) as f:
        tracks = xmltodict.parse(f.read())['annotations']['track']

    annotations = []
    for track in tracks:
        id = track['@id']
        label = track['@label']

        if label != 'car':
            continue

        for box in track['box']:
            annotations.append(BoundingBox(
                id=int(id),
                label=label,
                frame=int(box['@frame']),
                xtl=int(float(box['@xtl'])),
                ytl=int(float(box['@ytl'])),
                xbr=int(float(box['@xbr'])),
                ybr=int(float(box['@ybr'])),
                occluded=box['@occluded'] == '1',
                parked=box['attribute']['#text'] == 'true'
            ))

    return annotations

def read_detections(path):
    """
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """

    with open(path) as f:
        lines = f.readlines()

    detections = []
    for line in lines:
        det=line.split(sep=',')
        detections.append(BoundingBox(
            id=int(det[1]),
            label='car',
            frame=int(det[0])-1,
            xtl=int(float(det[2])),
            ytl=int(float(det[3])),
            xbr=int(float(det[2]))+int(float(det[4])),
            ybr=int(float(det[3]))+int(float(det[5])),
            confidence=float(det[6])
        ))

    return detections

def draw_boxes(image, frame_id, boxes, gt=False):
    if gt:
        color = (0,255,0)
    else:
        color = (255,0,0)
    for box in boxes:
        if box.frame == frame_id:
            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), color, 2)
            if not gt:
                cv2.putText(image, str(box.confidence), (box.xtl, box.ytl-5), cv2.FONT_ITALIC, 0.6, color, 2)
    return image

if __name__ == '__main__':
    annotations_path = './data/ai_challenge_s03_c010-full_annotation.xml'
    detections_path = './data/AICity_data/train/S03/c010/det/det_yolo3.txt'
    video_path = './data/AICity_data/train/S03/c010/vdo.avi'

    annotations = read_annotations(annotations_path)
    detections = read_detections(detections_path)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while(True):
        _, frame = cap.read()
        frame = draw_boxes(frame, frame_id, annotations, gt=True)
        frame = draw_boxes(frame, frame_id, detections)
        cv2.imshow('frame', frame)
        if(cv2.waitKey() == 113): #press q to quit
            break
        frame_id+=1