import argparse

import torch
from torch import nn

from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
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
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    return parser.parse_args()


# def compute_losses(net_output, jig_l, class_l):
#     return F.cross_entropy(net_output[0], jig_l), F.cross_entropy(net_output[1], class_l)

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        self.model = model.to(device)
        print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args.source, args.jigsaw_n_classes, val_size=args.val_size)
        self.target_loader = data_helper.get_val_dataloader(args.target, args.jigsaw_n_classes)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate)
        self.jig_weight = args.jig_weight
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
        else:
            self.target_id = len(args.source)

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)

            self.optimizer.zero_grad()

            jigsaw_logit, class_logit = self.model(data)
            jigsaw_loss = criterion(jigsaw_logit, jig_l)
            class_loss = criterion(class_logit[d_idx != self.target_id], class_l[d_idx != self.target_id])
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            loss = class_loss + jigsaw_loss * self.jig_weight

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader), {"jigsaw": jigsaw_loss.item(), "class": class_loss.item()},
                            {"jigsaw": torch.sum(jig_pred == jig_l.data).item(), "class": torch.sum(cls_pred == class_l.data).item()},
                            data.shape[0])
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                jigsaw_correct = 0
                class_correct = 0
                total = 0
                for it, ((data, jig_l, class_l), d_idx) in enumerate(loader):
                    data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
                    jigsaw_logit, class_logit = self.model(data)
                    _, cls_pred = class_logit.max(dim=1)
                    _, jig_pred = jigsaw_logit.max(dim=1)
                    class_correct += torch.sum(cls_pred == class_l.data)
                    jigsaw_correct += torch.sum(jig_pred == jig_l.data)
                    total += data.shape[0]
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_training(self):
        self.logger = Logger(self.args)
        self.results = {"val":torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
