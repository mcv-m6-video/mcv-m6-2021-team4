import os, torch, torchvision, random
from torch.utils.data import DataLoader
import matplotlib.image as pltImage
from collections import Counter
from collections import defaultdict
from pytorch_metric_learning.utils import common_functions

from siamese_network import trans_test, MLP

class Detections():
    def __init__(self, data, path, transform=trans_test):
        self.data = list(data.values[:, 0])
        self.box_id = list(data.values[:, 1])
        self.sequence = list(data.values[:, 2])
        self.camera = list(data.values[:, 3])
        self.frame_id = list(data.values[:, 4])
        self.xtl = list(data.values[:, 5])
        self.ytl = list(data.values[:, 6])
        self.xbr = list(data.values[:, 7])
        self.ybr = list(data.values[:, 8])
        self.center_x = list(data.values[:, 9])
        self.center_y = list(data.values[:, 10])
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index]
        label = self.box_id[index]
        img_path = os.path.join(self.path, (str(img_name)))
        image = pltImage.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)

        stats = [self.xtl[index], self.ytl[index],
                 abs(self.xbr[index] - self.xtl[index]),
                 abs(self.ybr[index] - self.ytl[index])]

        return image, label, self.camera[index], stats, self.frame_id[index]


def get_data_loader(split, labels, patches_path):
    df = labels[labels['FILENAME'].str.contains('S03')].sample(frac=1)
    data = Detections(df, patches_path, trans_test)
    loader = DataLoader(dataset=data, batch_size=32)
    return data, loader


def find_matches(test_data, id_frames_c1, frame_c2,
                 inference_model, patches_to_compare_c1):
    idx2_matches = {}
    for id_c1, frames_c1 in id_frames_c1.items():
        num_frames_to_compare_c1 = min(patches_to_compare_c1, len(frames_c1))
        for frame_c1 in random.sample(frames_c1, num_frames_to_compare_c1):
            x = torch.zeros(1, 3, 64, 64)
            y = torch.zeros(1, 3, 64, 64)
            x[0] = test_data[frame_c1][0]
            y[0] = test_data[frame_c2][0]
            if inference_model.is_match(x, y):
                idx2_matches[frame_c1] = id_c1

    return idx2_matches


def merge_dicts(dict1, dict2):
    for id, frames in dict2.items():
      if id in dict1:
        dict1[id].extend(frames)
      else:
        dict1[id] = frames
    return dict1


def compare_cams(test_data, id_frames_c1, id_frames_c2,
                 inference_model, patches_to_compare_c1,
                 patches_to_compare_c2):
    re_id_c2 = {}

    for id_c2, frames_c2 in id_frames_c2.items():
        idx2_matches = {}
        repeated_matches = {}
        num_frames_to_compare_c2 = min(patches_to_compare_c2, len(frames_c2))
        for frame_c2 in random.sample(frames_c2, num_frames_to_compare_c2):
            idx2_matches_tmp = find_matches(test_data, id_frames_c1, frame_c2,
                                    inference_model, patches_to_compare_c1)

            for frame, id in idx2_matches_tmp.items():
                if frame in idx2_matches:
                    if id in repeated_matches:
                        repeated_matches[id] += 1
                    else:
                        repeated_matches[id] = 1
                else:
                    idx2_matches[frame] = id

        if len(idx2_matches) == 0:
            re_id_c2[id_c2] = frames_c2
            continue

        freq_ids = dict(Counter(idx2_matches.values()))

        for i, c in repeated_matches.items():
            if i in freq_ids:
                freq_ids[i] += c

        max_freq_id = max(freq_ids, key=freq_ids.get)

        if max_freq_id in re_id_c2:
            re_id_c2[max_freq_id].extend(frames_c2)

        else:
            re_id_c2[max_freq_id] = frames_c2

    return re_id_c2


def get_id_frames_cam(test_data, indices_cameras, cam):
    indices_cam = indices_cameras[cam]
    test_data_cam = [[i, test_data[i][1]] for i in indices_cam]
    id_frames_cam = defaultdict(list)
    for key, value in sorted(dict(test_data_cam).items()):
        id_frames_cam[value].append(key)

    return id_frames_cam


def invert_dict(dict1):
  new_dict = {}
  for id, frames in dict1.items():
    for f in frames:
      new_dict[f] = id
  return new_dict


def load_trunk_embedder(trunk_path, embedder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set trunk model and replace the softmax layer with an identity function
    trunk = torchvision.models.resnet18(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = common_functions.Identity()
    trunk = torch.nn.DataParallel(trunk.to(device))

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))

    if device == 'cpu':
        trunk.load_state_dict(torch.load(trunk_path, map_location=torch.device('cpu')))
    else:
        trunk.load_state_dict(torch.load(trunk_path))
    embedder.load_state_dict(torch.load(embedder_path, map_location=torch.device('cpu')))

    return trunk, embedder