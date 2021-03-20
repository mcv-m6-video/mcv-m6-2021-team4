import xmltodict
from bounding_box import BoundingBox
from collections import defaultdict, OrderedDict

def group_by_frame(boxes):
    grouped = defaultdict(list)
    for box in boxes:
        grouped[box.frame].append(box)
    return OrderedDict(sorted(grouped.items()))

def read_annotations(path, grouped=True, use_parked=False):
    with open(path) as f:
        tracks = xmltodict.parse(f.read())['annotations']['track']

    annotations = []
    for track in tracks:
        id = track['@id']
        label = track['@label']

        # if label != 'car':
        #     continue

        if label == 'car':
            for box in track['box']:
                is_parked = box['attribute']['#text'] == 'true'
                if not use_parked and is_parked:
                    continue

                annotations.append(BoundingBox(
                    id=int(id),
                    label=label,
                    frame=int(box['@frame']),
                    xtl=float(box['@xtl']),
                    ytl=float(box['@ytl']),
                    xbr=float(box['@xbr']),
                    ybr=float(box['@ybr']),
                    occluded=box['@occluded'] == '1',
                    parked=is_parked
                ))
        elif label == 'bike':
            for box in track['box']:
                annotations.append(BoundingBox(
                id=int(id),
                label=label,
                frame=int(box['@frame']),
                xtl=float(box['@xtl']),
                ytl=float(box['@ytl']),
                xbr=float(box['@xbr']),
                ybr=float(box['@ybr']),
                occluded=box['@occluded'] == '1'
                ))

    if grouped:
        return group_by_frame(annotations)

    return annotations


def read_detections(path, grouped=False):
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

    if grouped:
        return group_by_frame(detections)

    return detections