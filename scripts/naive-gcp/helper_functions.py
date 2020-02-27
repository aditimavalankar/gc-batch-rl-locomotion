import numpy as np
import os
import pickle


def load_checkpoint(filename, net, optimizer):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    print('Loading checkpoint ', filename)
    checkpoint = torch.load(filename)
    epochs = int(checkpoint['epochs'])
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return epochs, net, optimizer
