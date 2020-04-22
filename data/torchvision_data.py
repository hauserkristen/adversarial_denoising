from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose

from .data_transforms import get_noise, ToRGBTensor, ScaleImage

def get_data(data_name: str, train: bool, noise_transform: str = '', percent_noise=1.0):
    # Define transform
    if data_name == 'MNIST':
        if noise_transform == '':
            trans = Compose([ToTensor()])
        else:
            trans = Compose([ToTensor(), get_noise(noise_transform, percent_noise)])
    else:
        if noise_transform == '':
            trans = Compose([ToTensor()])
        else:
            trans = Compose([ToRGBTensor(), get_noise(noise_transform, percent_noise), ScaleImage()])

    # Define download path
    data_path = 'data\\{}\\test\\'.format(data_name) if train else 'data\\{}\\train\\'.format(data_name)

    # Read data
    if data_name == 'MNIST':
        data_set = MNIST(data_path, train=train, download=True, transform=trans)
    elif data_name == 'CIFAR10':
        data_set = CIFAR10(data_path, train=train, download=True, transform=trans)
    else:
        raise NotImplementedError('Unkown data set: {}. '.format(data_name))

    return data_set