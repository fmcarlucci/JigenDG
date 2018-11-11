from time import time

from os.path import join, dirname

from .tf_logger import TFLogger

_log_path = join(dirname(__file__), '../logs')


# high level wrapper for tf_logger.TFLogger
class Logger():
    def __init__(self, args, update_frequency=10):
        self.current_epoch = 0
        self.max_epochs = args.epochs
        self.last_update = time()
        self.start_time = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        folder, logname = self.get_name_from_args(args)
        log_path = join(_log_path, folder, logname)
        if args.tf_logger:
            self.tf_logger = TFLogger(log_path)
            print("Saving to %s" % log_path)
        else:
            self.tf_logger = None
        self.current_iter = 0

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()
        if self.tf_logger:
            for n, v in enumerate(self.lrs):
                self.tf_logger.scalar_summary("aux/lr%d" % n, v, self.current_iter)

    def log(self, it, iters, losses, samples_right, total_samples):
        self.current_iter += 1
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / total_samples)) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.max_epochs, loss_string,
                                                                acc_string, total_samples))
            # update tf log
            if self.tf_logger:
                for k, v in losses.items(): self.tf_logger.scalar_summary("train/loss_%s" % k, v, self.current_iter)

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))
        if self.tf_logger:
            for k, v in accuracies.items(): self.tf_logger.scalar_summary("%s/acc_%s" % (phase, k), v, self.current_iter)

    def save_best(self, val_test, best_test):
        print("It took %g" % (time() - self.start_time))
        if self.tf_logger:
            for x in range(10):
                self.tf_logger.scalar_summary("best/from_val_test", val_test, x)
                self.tf_logger.scalar_summary("best/max_test", best_test, x)

    @staticmethod
    def get_name_from_args(args):
        folder_name = "%s_to_%s" % ("-".join(sorted(args.source)), args.target)
        if args.folder_name:
            folder_name = join(args.folder_name, folder_name)
        name = "eps%d_bs%d_lr%g_class%d_jigClass%d_jigWeight%g" % (args.epochs, args.batch_size, args.learning_rate, args.n_classes,
                                                                   args.jigsaw_n_classes, args.jig_weight)
        # if args.ooo_weight > 0:
        #     name += "_oooW%g" % args.ooo_weight
        if args.train_all:
            name += "_TAll"
        if args.bias_whole_image:
            name += "_bias%g" % args.bias_whole_image
        if args.classify_only_sane:
            name += "_classifyOnlySane"
        if args.TTA:
            name += "_TTA"
        try:
            name += "_entropy%g_jig_tW%g" % (args.entropy_weight, args.target_weight)
        except AttributeError:
            pass
        if args.suffix:
            name += "_%s" % args.suffix
        name += "_%d" % int(time() % 1000)
        return folder_name, name
