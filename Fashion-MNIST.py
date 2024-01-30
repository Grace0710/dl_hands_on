import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # plt.show()


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    training_set = torchvision.datasets.FashionMNIST('./data',
                                                     download=True,
                                                     train=True,
                                                     transform=transform)

    validation_set = torchvision.datasets.FashionMNIST('./data',
                                                       download=True,
                                                       train=False,
                                                       transform=transform)

    training_loader = torch.utils.data.DataLoader(training_set,
                                                  batch_size=4,
                                                  shuffle=True,
                                                  num_workers=2)

    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=4,
                                                    shuffle=False,
                                                    num_workers=2)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Extract a batch of 4 images
    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    writer.add_image('Four Fashion-MNIST Images', img_grid)
    writer.flush()