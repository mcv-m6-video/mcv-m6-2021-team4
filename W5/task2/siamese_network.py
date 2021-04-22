import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as pltImage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import  transforms

import umap
import record_keeper
from cycler import cycler

import pytorch_metric_learning
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import logging
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)


trans_train = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64)),
                            transforms.ToTensor(),
                        ])

trans_test = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64)),
                            transforms.ToTensor(),
                        ])


class CarsDataset(Dataset):
    def __init__(self, data, path, transform=trans_train):
        super().__init__()
        self.data = list(data.values[:, 0])
        self.targets = list(data.values[:, 1])
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index]
        label = self.targets[index]
        img_path = os.path.join(self.path, (str(img_name)))
        image = pltImage.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


def get_data_loader(split, labels, patches_path):
    if split == 'train':
        df = labels[labels['FILENAME'].str.contains('S01' or 'S04')].sample(frac=1)
        data = CarsDataset(df, patches_path, trans_train)
        loader = DataLoader(dataset=data, batch_size=32, shuffle=True)
    else:
        df = labels[labels['FILENAME'].str.contains('S03')].sample(frac=1)
        data = CarsDataset(df, patches_path, trans_test)
        loader = DataLoader(dataset=data, batch_size=32)
    return data, loader


def train(train_data, test_data, save_model, num_epochs, lr, embedding_size, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set trunk model and replace the softmax layer with an identity function
    trunk = torchvision.models.resnet18(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = common_functions.Identity()
    trunk = torch.nn.DataParallel(trunk.to(device))

    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder = torch.nn.DataParallel(MLP([trunk_output_size, embedding_size]).to(device))

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=lr/10, weight_decay=0.0001)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=lr, weight_decay=0.0001)

    # Set the loss function
    loss = losses.TripletMarginLoss(margin=0.1)

    # Set the mining function
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_data.targets, m=4, length_before_new_iter=len(train_data))

    save_dir = os.path.join(save_model,
                            ''.join(str(lr).split('.')) + '_' + str(batch_size) + '_' + str(embedding_size))

    os.makedirs(save_dir, exist_ok=True)

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}

    record_keeper, _, _ = logging_presets.get_record_keeper(os.path.join(save_dir, "example_logs"),
                                                            os.path.join(save_dir, "example_tensorboard"))
    hooks = logging_presets.get_hook_container(record_keeper)

    dataset_dict = {"val": test_data, "train": train_data}
    model_folder = "example_saved_models"

    def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
        logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
        label_set = np.unique(labels)
        num_classes = len(label_set)
        fig = plt.figure(figsize=(20, 15))
        plt.title(str(split_name) + '_' + str(num_embeddings))
        plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
        plt.show()

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook,
                                                visualizer=umap.UMAP(),
                                                visualizer_hook=visualizer_hook,
                                                dataloader_num_workers=32,
                                                accuracy_calculator=AccuracyCalculator(k="max_bin_count"))

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                                dataset_dict,
                                                model_folder,
                                                test_interval=1,
                                                patience=1)

    trainer = trainers.MetricLossOnly(models,
                                      optimizers,
                                      batch_size,
                                      loss_funcs,
                                      mining_funcs,
                                      train_data,
                                      sampler=sampler,
                                      dataloader_num_workers=32,
                                      end_of_iteration_hook=hooks.end_of_iteration_hook,
                                      end_of_epoch_hook=end_of_epoch_hook)

    trainer.train(num_epochs=num_epochs)

    if save_model is not None:

        torch.save(models["trunk"].state_dict(), os.path.join(save_dir, 'trunk.pth'))
        torch.save(models["embedder"].state_dict(), os.path.join(save_dir, 'embedder.pth'))

        print('Model saved in ', save_dir)
