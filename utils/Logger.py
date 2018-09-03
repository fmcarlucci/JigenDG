from time import time


class Logger():
    def __init__(self, max_epochs, update_frequency=10):
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.last_update = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        self.losses = {"jigsaw": [], "class": []}
        self.val_acc = []

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()

    def log(self, it, iters, losses, samples_right, total_samples):
        loss_string = ", ".join(["%s : %f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %f" % (k, v / total_samples) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.max_epochs, loss_string,
                                                                acc_string, total_samples))
        for k, v in losses.items():
            self.losses[k].append(v)

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, accuracies):
        print("Accuracies on target: " + ", ".join(["%s : %f" % (k, v) for k, v in accuracies.items()]))
        self.val_acc.append(accuracies["class"])
