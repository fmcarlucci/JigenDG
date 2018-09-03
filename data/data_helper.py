from os.path import join, dirname

import torch
from torch.utils.data import DataLoader

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset
from data.concat_dataset import ConcatDataset

available_datasets = ["amazon", "dslr", "webcam"]
paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in available_datasets}


def get_dual_dataloaders(dataset_names, jig_classes, phase, batch_size=128):
    assert isinstance(dataset_names, list)
    if phase == "train":
        print("Using multiple sources")
        jig_dataset = ConcatDataset(
            [JigsawDataset("", join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), patches=False, classes=jig_classes) for dname in dataset_names])
        standard_dataset = StandardDataset.get_dataset(paths[dataset_names[0]], phase, 227)
        jig_dataloader = DataLoader(jig_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        standard_dataloader = DataLoader(standard_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        return jig_dataloader, standard_dataloader
    else:
        standard_dataset = StandardDataset.get_dataset(paths[dataset_names[0]], phase, 227)
        return DataLoader(standard_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)


def get_dataloader(dataset_name, jig_classes, phase, batch_size=128):
    if phase == "train":
        shuffle = True
        drop_last = True
        data_func = JigsawDataset
    else:
        shuffle = False
        drop_last = False
        data_func = JigsawTestDataset
    if isinstance(dataset_name, list):
        print("Using multiple sources")
        dataset = ConcatDataset(
            [data_func("", join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), patches=False, classes=jig_classes) for dname in dataset_name])
    else:
        dataset = ConcatDataset([data_func("", join(dirname(__file__), 'txt_lists', '%s_train.txt' % dataset_name), patches=False, classes=jig_classes)])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=drop_last)
