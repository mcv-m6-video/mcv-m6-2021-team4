import numpy as np
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_iou
import imageio
import os

def task2(gt_path, det_path, video_path,results_path):

    plot_frames_path = os.path.join(results_path, '/plot_frames/')
    video_frames_path = os.path.join(results_path, '/video_frames/')

    # If folder  doesn't exist -> create it
    if not os.path.exists(plot_frames_path):
        os.makedirs(plot_frames_path)

    if not os.path.exists(video_frames_path):
        os.makedirs(video_frames_path)

    show_det = True
    show_noisy = False

    gt = read_annotations(gt_path)
    det = read_detections(det_path)

    grouped_gt = group_by_frame(gt)
    grouped_det = group_by_frame(det)

    noise_params = {
        'add': False,
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


        plot = plot_iou(iou_list,num_frames)
        plot_image = plot_to_image(plot)



        '''
        if show:
            fig.show()
            cv2.imshow('frame', frame)
            if cv2.waitKey() == 113:  # press q to quit
                break
        '''
        imageio.imwrite(video_frames_path+'{}.png'.format(frame_id), frame)
        plot.savefig(plot_frames_path+'iou_{}.png'.format(frame_id))
        plt.close(plot)


        frame_id += 1
    save_gif(plot_frames_path,results_path+'iou.gif')
    save_gif(video_frames_path,results_path+'bbox.gif')
    #cv2.destroyAllWindows()

    return


def save_gif(source_path,results_path):
    # Build GIF

    with imageio.get_writer(results_path, mode='I') as writer:
        for filename in sorted(os.listdir(source_path)):
            image = imageio.imread(filename)
            writer.append_data(image)


def plot_iou(dict_iou,xmax):
    lists = sorted(dict_iou.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x, y)
    ax.grid()
    ax.set(xlabel='frame', ylabel='IoU',
           title='IoU vs Time')

    # Used to keep the limits constant
    ax.set_ylim(0,1)
    ax.set_xlim(0,xmax)

    return fig

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

    