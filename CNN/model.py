from torch import nn
from torch.nn import Sequential, Conv2d, Linear, Tanh, Flatten, Dropout


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = Sequential(
            Conv2d(1, 10, (10, 1), stride=1, padding='same', padding_mode='zeros'),
            Tanh(),

            Conv2d(10, 10, (10, 1), stride=1, padding='same', padding_mode='zeros'),
            Tanh(),

            Conv2d(10, 10, (10, 1), stride=1, padding='same', padding_mode='zeros'),
            Tanh(),

            Conv2d(10, 10, (10, 1), stride=1, padding='same', padding_mode='zeros'),
            Tanh(),

            Conv2d(10, 1, (3, 1), stride=1, padding='same', padding_mode='zeros'),
            Tanh(),

            Flatten(),
            Tanh(),
            Dropout()

        )
        self.fn = Linear(420, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fn(x)

        return x
