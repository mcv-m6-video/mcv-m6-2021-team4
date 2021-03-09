import numpy as np
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_iou


def task2(gt_path, det_path, video_path):
    #video_path = '../../data/AICity_data/train/S03/c010/vdo.avi'

    show_det = False
    show_noisy = True

    gt = read_annotations(gt_path)
    det = read_detections(det_path)

    grouped_gt = group_by_frame(gt)
    grouped_det = group_by_frame(det)

    noise_params = {
        'add': True,
        'drop': 0.0,
        'generate_close': 0.0,
        'generate_random': 0.0,
        'type': 'specific',  # options: 'specific', 'gaussian', None
        'std': 40,  # pixels
        'position': False,
        'size': True,
        'keep_ratio': True
    }

    # to generate a BB randomly in an image, we use the mean width and
    # height of the annotated BBs so that they have similar statistics
    if noise_params['generate_random'] > 0.0:
        mean_w = 0
        mean_h = 0
        for b in gt:
            mean_w += b.width
            mean_h += b.height
        mean_w /= len(gt)
        mean_h /= len(gt)

    # if we want to replicate results
    # np.random.seed(10)

    if noise_params['add']:
        noisy_gt = add_noise(gt, noise_params)
        grouped_noisy_gt = group_by_frame(noisy_gt)

    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



    frame_list = []
    iou_plots = []
    iou_list = {}

    for frame_id in range(num_frames):
        _, frame = cap.read()

        frame = draw_boxes(frame, frame_id, grouped_gt[frame_id], color='g')

        if show_det:
            frame = draw_boxes(frame, frame_id, grouped_det[frame_id], color='b', det=True)
            frame_iou = mean_iou(grouped_det[frame_id],grouped_gt[frame_id],sort=True)

        if show_noisy:
            frame = draw_boxes(frame, frame_id, grouped_noisy_gt[frame_id], color='r')
            frame_iou = mean_iou(grouped_noisy_gt[frame_id],grouped_gt[frame_id])

        iou_list[frame_id] = frame_iou


        '''
        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:  # press q to quit
            break
        '''

        frame_list.append(frame)


        frame_id += 1

    cv2.destroyAllWindows()

    return


def save_gif():



def plot_iou(dict_iou):
    lists = sorted(dict_iou.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.ylim(0,1)
    plt.xlim(0,len(lists))

    plt.show()

def mean_iou(det,gt,sort=False):
    '''
    det: detections of one frame
    gt: annotations of one frame
    sort: False if we use modified GT, 
          True if we have a confidence value for the detection
    '''
    if sort:
        BB = sort_by_confidence(det)
    else:
        BB = np.array([x.box for x in det]).reshape(-1, 4)    

    BBGT = np.array([anot.box for anot in gt])
    
    
    nd = len(BB)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    mean_iou=[]
    for d in range(nd):
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
    
        if BBGT.size > 0:
            # compute overlaps
            overlaps = voc_iou(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            mean_iou.append(ovmax)

    return np.mean(mean_iou)


def sort_by_confidence(det):
    BB = np.array([x.box for x in det]).reshape(-1, 4)
    confidence = np.array([float(x.confidence) for x in det])
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    return BB

if __name__ == '__main__':

    gt_path = '/Data/ai_challenge_s03_c010-full_annotation.xml'
    det_path = '/Data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt'

    gt = read_annotations(gt_path)
    det = read_detections(det_path)

    annotations_grouped = group_by_frame(gt)

    # read annotations
    class_recs = {}
    npos = 0

    for frame_id, boxes in annotations.items():
        bbox = np.array([det.box for det in boxes])
        det = [False] * len(boxes)
        npos += len(boxes)
        class_recs[frame_id] = {"bbox": bbox, "det": det}

    # read detections
    image_ids = [x.frame for x in det]
    BB = np.array([x.box for x in det]).reshape(-1, 4)

    confidence = np.array([float(x.confidence) for x in det])
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down detections (dets) and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    mean_iou = []
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = voc_iou(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            iou = np.mean(np.max(overlaps))
            mean_iou.append(iou)