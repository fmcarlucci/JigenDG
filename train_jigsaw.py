import argparse

import torch
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=False, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    return parser.parse_args()


# def compute_losses(net_output, jig_l, class_l):
#     return F.cross_entropy(net_output[0], jig_l), F.cross_entropy(net_output[1], class_l)

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args.source, args.jigsaw_n_classes, val_size=args.val_size, batch_size=args.batch_size,
                                                                               bias_whole_image=args.bias_whole_image, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args.target, args.jigsaw_n_classes, multi=args.TTA, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all)
        self.jig_weight = args.jig_weight
        self.ooo_weight = args.ooo_weight
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
        else:
            self.target_id = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, jig_l, class_l, ooo), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, ooo = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), ooo.to(self.device)

            self.optimizer.zero_grad()

            jigsaw_logit, class_logit, ooo_logit = self.model(data)
            jigsaw_loss = criterion(jigsaw_logit, jig_l)
            ooo_loss = criterion(ooo_logit, ooo)
            if self.only_non_scrambled:
                class_loss = criterion(class_logit[jig_l == 0], class_l[jig_l == 0])
            elif self.target_id:
                class_loss = criterion(class_logit[d_idx != self.target_id], class_l[d_idx != self.target_id])
            else:
                class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, ooo_pred = ooo_logit.max(dim=1)
            loss = class_loss + jigsaw_loss * self.jig_weight  + ooo_loss * self.ooo_weight

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader), 
                            {"jigsaw": jigsaw_loss.item(), "class": class_loss.item(), "OOO": ooo_loss.item()},
                            {"jigsaw": torch.sum(jig_pred == jig_l.data).item(), 
                             "class": torch.sum(cls_pred == class_l.data).item(),
                             "OOO": torch.sum(ooo_pred == ooo.data).item()},
                            data.shape[0])
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                if loader.dataset.isMulti():
                    jigsaw_correct, class_correct, single_acc = self.do_test_multi(loader)
                    print("Single vs multi: %g %g" % (float(single_acc) / total, float(class_correct) / total))
                else:
                    jigsaw_correct, class_correct, ooo_correct = self.do_test(loader)
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                ooo_acc = float(ooo_correct) / total
                self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc, "OOO": ooo_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        ooo_correct = 0
        for it, ((data, jig_l, class_l, ooo_l), d_idx) in enumerate(loader):
            data, jig_l, class_l, ooo_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), ooo_l.to(self.device)
            jigsaw_logit, class_logit, ooo_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, ooo_pred = ooo_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
            ooo_correct += torch.sum(ooo_pred == ooo_l.data)
        return jigsaw_correct, class_correct, ooo_correct

    def do_test_multi(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        single_correct = 0
        for it, ((data, jig_l, class_l), d_idx) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            n_permutations = data.shape[1]
            class_logits = torch.zeros(n_permutations, data.shape[0], self.n_classes).to(self.device)
            for k in range(n_permutations):
                class_logits[k] = F.softmax(self.model(data[:, k])[1], dim=1)
            class_logits[0] *= 4 * n_permutations  # bias more the original image
            class_logit = class_logits.mean(0)
            _, cls_pred = class_logit.max(dim=1)
            jigsaw_logit, single_logit = self.model(data[:, 0])
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, single_logit = single_logit.max(dim=1)
            single_correct += torch.sum(single_logit == class_l.data)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data[:, 0])
        return jigsaw_correct, class_correct, single_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
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
