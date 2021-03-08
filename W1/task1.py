import xmltodict
import cv2
import numpy as np
from copy import deepcopy

img_shape = [1080, 1920]

colors = {
    'r': (0, 0, 255),
    'g': (0, 255, 0),
    'b': (255, 0, 0)
}


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

    @property
    def box(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def center(self):
        return [(self.xtl + self.xbr) // 2, (self.ytl + self.ybr) // 2]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ybr - self.ytl)

    def area(self):
        return self.width * self.height

    def shift_position(self, center):
        self.xtl = int(center[0] - self.width / 2)
        self.ytl = int(center[1] - self.height / 2)
        self.xbr = int(center[0] + self.width / 2)
        self.ybr = int(center[1] + self.height / 2)
        return

    def resize(self, size):

        # check if width and height > 0?

        h, w = size
        c = self.center
        self.xtl = int(c[0] - w / 2)
        self.ytl = int(c[1] - h / 2)
        self.xbr = int(c[0] + w / 2)
        self.ybr = int(c[1] + h / 2)
        return

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
        det = line.split(sep=',')
        detections.append(BoundingBox(
            id=int(det[1]),
            label='car',
            frame=int(det[0]) - 1,
            xtl=int(float(det[2])),
            ytl=int(float(det[3])),
            xbr=int(float(det[2])) + int(float(det[4])),
            ybr=int(float(det[3])) + int(float(det[5])),
            confidence=float(det[6])
        ))

    return detections


def draw_boxes(image, frame_id, boxes, color='g', det=False):
    rgb = colors[color]
    for box in boxes:
        if box.frame == frame_id:
            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), rgb, 2)
            if det:
                cv2.putText(image, str(box.confidence), (box.xtl, box.ytl - 5), cv2.FONT_ITALIC, 0.6, rgb, 2)
    return image


def add_specific_noise(box, noise_params):
    if noise_params['position']:
        noisy_center = box.center + np.random.normal(0, noise_params['std'], 2)
        box.shift_position(noisy_center)

    if noise_params['size']:
        # generate this random width/height proportional to the initial width/height? (e.g. width * [0.95,1.05])
        if noise_params['keep_ratio']:
            size_noise = np.random.normal(0, noise_params['std'], 1)
            h = box.height + size_noise[0]
        else:
            size_noise = np.random.normal(0, noise_params['std'], 2)
            h = box.height + size_noise[1]

        w = box.width + size_noise[0]
        box.resize([h, w])

    return


def add_gaussian_noise(box, noise_params):
    noisy_coords = np.random.normal(0, noise_params['std'], 4)
    box.xtl = int(box.xtl + noisy_coords[0])
    box.ytl = int(box.ytl + noisy_coords[1])
    box.xbr = int(box.xbr + noisy_coords[2])
    box.ybr = int(box.ybr + noisy_coords[3])

    return


def add_noise(annotations, noise_params):
    noisy_annotations = []
    for box in annotations:

        # remove box
        if np.random.random() <= noise_params['drop']:
            continue

        # generate box
        if np.random.random() <= noise_params['generate']:
            new_box = deepcopy(box)
            noise['gaussian'](new_box, noise_params)  # add random noise to the new box (specify another std?)
            noisy_annotations.append(new_box)

        b = deepcopy(box)
        # add noise to existing box
        if noise_params['type'] is not None:
            noise[noise_params['type']](b, noise_params)

        noisy_annotations.append(b)

    return noisy_annotations


noise = {
    'gaussian': add_gaussian_noise,
    'specific': add_specific_noise
}

if __name__ == '__main__':
    annotations_path = '../../data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'
    detections_path = '../../data/AICity_data/train/S03/c010/det/det_yolo3.txt'
    video_path = '../../data/AICity_data/train/S03/c010/vdo.avi'

    # argparse ?

    show_annotations = True
    show_detections = False
    show_noisy = True

    noise_params = {
        'add': True,  # tmp name
        'drop': 0.0,
        'generate': 0.0,
        'type': None,
        'std': 40,
        'position': False,
        'size': True,
        'keep_ratio': True
    }

    annotations = read_annotations(annotations_path)
    detections = read_detections(detections_path)

    # is this necessary? we could compute the w and h randomly as well
    mean_w = 0
    mean_h = 0
    for b in annotations:
        mean_w += b.width
        mean_h += b.height
    mean_w /= len(annotations)
    mean_h /= len(annotations)

    # np.random.seed(10)

    if noise_params['add']:
        noisy_annotations = add_noise(annotations, noise_params)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prob_generate = 0.3

    while True:
        _, frame = cap.read()

        if noise_params['add'] and np.random.random() <= prob_generate:
            cx = np.random.randint(0, img_shape[1])
            cy = np.random.randint(0, img_shape[0])
            w = np.random.normal(mean_w, 50)
            h = np.random.normal(mean_h, 50)
            noisy_annotations.append(BoundingBox(
                id=-1,
                label='car',
                frame=frame_id,
                xtl=int(cx - w/2),
                ytl=int(cy - h/2),
                xbr=int(cx + w/2),
                ybr=int(cy + h/2)
            ))

        if show_annotations:
            frame = draw_boxes(frame, frame_id, annotations, color='g')

        if show_detections:
            frame = draw_boxes(frame, frame_id, detections, color='b', det=True)

        if show_noisy:
            frame = draw_boxes(frame, frame_id, noisy_annotations, color='r')

        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:  # press q to quit
            break

        frame_id += 1
