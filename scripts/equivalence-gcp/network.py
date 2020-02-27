import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Encoder, self).__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        modules = []
        modules.append(nn.Linear(input_shape, 256))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(256, 256))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(256, output_shape))
        modules.append(nn.ReLU())
        self.sequential = nn.Sequential(*modules)
        self.sequential.apply(init_weights)

    def forward(self, x):
        return self.sequential(x)


class Policy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Policy, self).__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        modules = []
        modules.append(nn.Linear(input_shape, 50))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(50, 50))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(50, output_shape))
        modules.append(nn.Tanh())
        self.sequential = nn.Sequential(*modules)
        self.sequential.apply(init_weights)

    def forward(self, x):
        return self.sequential(x)
