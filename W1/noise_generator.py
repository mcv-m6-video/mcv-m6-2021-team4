import numpy as np
from copy import deepcopy

def add_specific_noise(box, noise_params):
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


def add_gaussian_noise(box, noise_params):
    noisy_coords = np.random.normal(0, noise_params['std'], 4)
    box.xtl = box.xtl + noisy_coords[0]
    box.ytl = box.ytl + noisy_coords[1]
    box.xbr = box.xbr + noisy_coords[2]
    box.ybr = box.ybr + noisy_coords[3]

    return


def add_noise(annotations, noise_params):
    noisy_annotations = []
    for box in annotations:

        # remove box
        if np.random.random() <= noise_params['drop']:
            continue

        # generate box
        if np.random.random() <= noise_params['generate_close']:
            new_box = deepcopy(box)
            noise['gaussian'](new_box, noise_params)
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