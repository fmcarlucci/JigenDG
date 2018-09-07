from torch import optim


def get_optim_and_scheduler(network, epochs, lr):
    optimizer = optim.SGD(network.get_params(lr), weight_decay=.0005, momentum=.9, nesterov=True, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler
