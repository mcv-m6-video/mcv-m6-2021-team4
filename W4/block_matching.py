import cv2
import numpy as np
from tqdm import tqdm


def distance(blockA, blockB, distance_metric='ncc'):
    if distance_metric == 'sad':
        return np.sum(np.abs(blockA-blockB))

    elif distance_metric == 'ssd':
        return np.sum((blockA-blockB)**2)

    elif distance_metric == 'ncc':
        return -cv2.matchTemplate(blockA, blockB, method=cv2.TM_CCORR_NORMED).squeeze()

    else:
        raise ValueError('Unknown distance metric')


def estimate_flow_block(N, distance_metric, blocks_positions, img_reference, img_current):
    tlx_ref = blocks_positions['tlx_ref']
    tly_ref = blocks_positions['tly_ref']
    init_tlx_curr = blocks_positions['init_tlx_curr']
    init_tly_curr = blocks_positions['init_tly_curr']
    end_tlx_curr = blocks_positions['end_tlx_curr']
    end_tly_curr = blocks_positions['end_tly_curr']

    if distance_metric == 'ncc':
        corr = cv2.matchTemplate(img_current[init_tly_curr:end_tly_curr + N, init_tlx_curr:end_tlx_curr + N],
                                 img_reference[tly_ref:tly_ref + N, tlx_ref:tlx_ref + N],
                                 method=cv2.TM_CCORR_NORMED)

        motion = list(np.unravel_index(np.argmax(corr), corr.shape))
        motion[0] = motion[0] + init_tly_curr - tly_ref
        motion[1] = motion[1] + init_tlx_curr - tlx_ref

        return [motion[1], motion[0]]

    else:
        min_dist = np.inf

        for tly_curr in np.arange(init_tly_curr, end_tly_curr):
            for tlx_curr in np.arange(init_tlx_curr, end_tlx_curr):

                dist = distance(img_reference[tly_ref:tly_ref + N, tlx_ref:tlx_ref + N],
                                img_current[tly_curr:tly_curr + N, tlx_curr:tlx_curr + N],
                                distance_metric)

                if dist < min_dist:
                    min_dist = dist
                    motion = [tlx_curr - tlx_ref, tly_curr - tly_ref]

        return motion


def estimate_flow(motion_type, N, P, distance_metric, img_reference, img_current):
    h, w = img_reference.shape[:2]
    flow = np.zeros(shape=(h, w, 2))

    for tly_ref in tqdm(range(0, h - N, N), desc='Motion estimation'):
        for tlx_ref in range(0, w - N, N):

            blocks_positions = {
                'tlx_ref': tlx_ref,
                'tly_ref': tly_ref,
                'init_tlx_curr': max(tlx_ref - P, 0),
                'init_tly_curr': max(tly_ref - P, 0),
                'end_tlx_curr': min(tlx_ref + P, w - N),
                'end_tly_curr': min(tly_ref + P, h - N)
            }

            flow[tly_ref:tly_ref + N, tlx_ref:tlx_ref + N, :] = estimate_flow_block(N, distance_metric, blocks_positions, img_reference, img_current)

    if motion_type == 'backward':
        flow = -flow

    return flow