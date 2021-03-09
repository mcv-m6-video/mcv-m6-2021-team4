import numpy as np
from copy import deepcopy
from bounding_box import BoundingBox

img_shape = [1080, 1920]


def add_specific_noise_box(box, noise_params):
    if noise_params['position']:
        noisy_center = box.center + np.random.normal(0, noise_params['std'], 2)
        box.shift_position(noisy_center)

    if noise_params['size']:
        if noise_params['keep_ratio']:
            size_noise = np.random.normal(0, noise_params['std'], 1)
            h = box.height + size_noise[0]
        else:
            size_noise = np.random.normal(0, noise_params['std'], 2)
            h = box.height + size_noise[1]

        w = box.width + size_noise[0]
        box.resize([h, w])

    return


def add_gaussian_noise_box(box, noise_params):
    noisy_coords = np.random.normal(0, noise_params['std'], 4)
    box.xtl = box.xtl + noisy_coords[0]
    box.ytl = box.ytl + noisy_coords[1]
    box.xbr = box.xbr + noisy_coords[2]
    box.ybr = box.ybr + noisy_coords[3]

    return


def add_noise(annotations, noise_params, num_frames):
    # to generate a BB randomly in an image, we use the mean width and
    # height of the annotated BBs so that they have similar statistics
    mean_w = 0
    mean_h = 0

    noisy_annotations = []
    # add noise to existing BBs (annotations)
    for box in annotations:
        # remove BB
        if np.random.random() <= noise_params['drop']:
            continue

        # generate BB close to an existing one
        if np.random.random() <= noise_params['generate_close']:
            new_box = deepcopy(box)
            noise['gaussian'](new_box, noise_params)
            noisy_annotations.append(new_box)

        b = deepcopy(box)
        # add noise to existing BB
        if noise_params['type'] is not None:
            noise[noise_params['type']](b, noise_params)

        noisy_annotations.append(b)

        mean_w += box.width
        mean_h += box.height

    mean_w /= len(annotations)
    mean_h /= len(annotations)

    # generate random BBs in random frames
    for frame_id in range(num_frames):
        if np.random.random() <= noise_params['generate_random']:
            # center of the BB (cx, cy), width and height (w, h)
            cx = np.random.randint(mean_w // 2, img_shape[1] - mean_w // 2)
            cy = np.random.randint(mean_h // 2, img_shape[0] - mean_h // 2)
            w = np.random.normal(mean_w, 10)
            h = np.random.normal(mean_h, 10)
            noisy_annotations.append(BoundingBox(
                id=-1,
                label='car',
                frame=frame_id,
                xtl=cx - w / 2,
                ytl=cy - h / 2,
                xbr=cx + w / 2,
                ybr=cy + h / 2
            ))

    return noisy_annotations


noise = {
    'gaussian': add_gaussian_noise_box,
    'specific': add_specific_noise_box
}