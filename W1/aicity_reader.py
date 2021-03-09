import xmltodict
from bounding_box import BoundingBox
from collections import defaultdict, OrderedDict

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
                xtl=float(box['@xtl']),
                ytl=float(box['@ytl']),
                xbr=float(box['@xbr']),
                ybr=float(box['@ybr']),
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
            xtl=float(det[2]),
            ytl=float(det[3]),
            xbr=float(det[2]) + float(det[4]),
            ybr=float(det[3]) + float(det[5]),
            confidence=float(det[6])
        ))

    return detections


def group_by_frame(boxes):
    grouped = defaultdict(list)
    for box in boxes:
        grouped[box.frame].append(box)
    return OrderedDict(sorted(grouped.items()))