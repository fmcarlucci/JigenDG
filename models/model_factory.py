from models import caffenet
from models import patch_based
from models import alexnet

nets_map = {
    'caffenet': caffenet.caffenet,
    'caffenet_jigsaw': patch_based.caffenet_patches,
    'alexnet': alexnet.alexnet
}


def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
