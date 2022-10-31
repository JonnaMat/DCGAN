import torch.nn as nn


FEATURE_MAP_SIZE = 48
# we are only working with RGB images
CHANNELS = 3


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(CHANNELS, FEATURE_MAP_SIZE, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(FEATURE_MAP_SIZE, FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURE_MAP_SIZE * 2, FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(FEATURE_MAP_SIZE * 4, FEATURE_MAP_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURE_MAP_SIZE * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(84, FEATURE_MAP_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_SIZE * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                FEATURE_MAP_SIZE * 8, FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(FEATURE_MAP_SIZE * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                FEATURE_MAP_SIZE * 4, FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(FEATURE_MAP_SIZE * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                FEATURE_MAP_SIZE * 2, FEATURE_MAP_SIZE, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(FEATURE_MAP_SIZE),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAP_SIZE, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


def initialize_weights(model):
    """
    Randomly initialize all weights to mean=0 and std=0.02 as proposed in the DCGAN paper
    `Unsupervised representative learning with deep convolutional generative adversarial networks`
    by Radford et al.
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def create_dcgan(device):
    """Create and initialize generator and discriminator."""
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    initialize_weights(netG)
    initialize_weights(netD)

    # Print the models
    print(netG)
    print(netD)

    return netD, netG
