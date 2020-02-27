import numpy as np
import os
import pickle
import torch
import errno


def load_checkpoint(ckpt_file,
                    latent_net,
                    control_net,
                    optimizer):

    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                ckpt_file)

    print('Loading checkpoint', ckpt_file)

    ckpt = torch.load(ckpt_file)
    epochs = int(ckpt['epochs'])
    latent_net.load_state_dict(ckpt['latent_state_dict'])
    control_net.load_state_dict(ckpt['control_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return epochs, latent_net, control_net, optimizer
