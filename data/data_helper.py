from os.path import join, dirname

import torch
from torch.utils.data import DataLoader

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets
office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
paths = {**office_paths, **pacs_paths, **vlcs_paths}

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                }


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(dataset_list, jig_classes, batch_size=128, image_size=225,
                         val_size=0.0, bias_whole_image=None, patches=False, limit=None, tune=False):
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), val_size)
        train_dataset = JigsawDataset(name_train, labels_train, patches=patches, image_size=image_size,
                                      classes=jig_classes, bias_whole_image=bias_whole_image)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        val_datasets.append(JigsawTestDataset(name_val, labels_val, patches=patches, image_size=image_size, classes=jig_classes))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_dataloader(dataset_name, jig_classes, batch_size=128, image_size=225, patches=False, tune=False):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % dataset_name))
    val_dataset = JigsawTestDataset(names, labels, patches=patches, image_size=image_size, classes=jig_classes)
    if len(val_dataset) > 30000:
        val_dataset = Subset(val_dataset, 30000)
        print("Using 30000 subset of val dataset")
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader
