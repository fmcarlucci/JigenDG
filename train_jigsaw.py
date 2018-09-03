import argparse

import torch

from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from torch.nn import functional as F
from torch import nn

from utils.Logger import Logger


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source")
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    return parser.parse_args()


# def compute_losses(net_output, jig_l, class_l):
#     return F.cross_entropy(net_output[0], jig_l), F.cross_entropy(net_output[1], class_l)


def do_epoch(model, source, target, optimizer, logger, device):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for it, (data, jig_l, class_l) in enumerate(source):
        data, jig_l, class_l = data.to(device), jig_l.to(device), class_l.to(device)
        
        optimizer.zero_grad()
        
        jigsaw_logit, class_logit = model(data)
        jigsaw_loss = criterion(jigsaw_logit, jig_l)
        class_loss = criterion(class_logit, class_l)
        _, cls_pred = class_logit.max(dim=1)
        _, jig_pred = jigsaw_logit.max(dim=1)
        loss = class_loss + jigsaw_loss * jig_weight
        
        loss.backward()
        optimizer.step()

        logger.log(it, len(source), {"jigsaw": jigsaw_loss, "class": class_loss},
                  {"jigsaw": torch.sum(jig_pred == jig_l.data).item(), "class":torch.sum(cls_pred == class_l.data).item()},
                  data.shape[0])

    model.eval()
    with torch.no_grad():
        jigsaw_correct = 0
        class_correct = 0
        total = 0
        for it, (data, jig_l, class_l) in enumerate(target):
            data, jig_l, class_l = data.to(device), jig_l.to(device), class_l.to(device)
            jigsaw_logit, class_logit = model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
            total += data.shape[0]
        logger.log_test({"jigsaw": float(jigsaw_correct) / total,
                         "class": float(class_correct) / total})


def do_training(epochs, model, source, target, optimizer, scheduler, device):
    logger = Logger(epochs)
    for k in range(epochs):
        scheduler.step()
        logger.new_epoch(scheduler.get_lr())
        do_epoch(model, source, target, optimizer, logger, device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes, classes=args.n_classes)
    print(model)
    model = model.to(device)
    source_dataloader = data_helper.get_dataloader(args.source, args.n_classes, "train")
    target_dataloader = data_helper.get_dataloader(args.target, args.n_classes, "val")
    optimizer, scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate)
    do_training(args.epochs, model, source_dataloader, target_dataloader, optimizer, scheduler, device)
    
    
if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.benchmark = True
    jig_weight = args.jig_weight
    main(args)
