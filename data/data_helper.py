from os.path import join, dirname

import torch
from torch.utils.data import DataLoader

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset

IMAGE_SIZE = 224

pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
available_datasets = office_datasets + pacs_datasets
office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
paths = {**office_paths, **pacs_paths}


def get_dual_dataloaders(dataset_names, jig_classes, phase, batch_size=128):
    if phase == "train":
        assert isinstance(dataset_names, list)
        print("Using multiple sources")
        jig_dataset = ConcatDataset(
            [JigsawDataset("", join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), patches=False, classes=jig_classes) for dname in dataset_names])
        standard_dataset = StandardDataset.get_dataset(paths[dataset_names[0]], phase, IMAGE_SIZE)
        jig_dataloader = DataLoader(jig_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        standard_dataloader = DataLoader(standard_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        return jig_dataloader, standard_dataloader
    else:
        standard_dataset = StandardDataset.get_dataset(paths[dataset_names], phase, IMAGE_SIZE)
        return DataLoader(standard_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)


def get_train_dataloader(dataset_name, jig_classes, batch_size=128, val_size=0.0):
    datasets = []
    val_datasets = []
    for dname in dataset_name:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), val_size)
        datasets.append(JigsawDataset(name_train, labels_train, patches=False, classes=jig_classes))
        val_datasets.append(JigsawDataset(name_val, labels_val, patches=False, classes=jig_classes))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset[val_datasets]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_train_dataloader(dataset_name, jig_classes, batch_size=128):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dataset_name))
    dataset = ConcatDataset([JigsawTestDataset(names, labels, patches=False, classes=jig_classes)])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader
