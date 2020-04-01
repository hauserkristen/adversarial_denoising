from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

from .data_transforms import get_noise

def get_data(data_name: str, train: bool, noise_transform: str = '', percent_noise=1.0):
    # Define transform
    if noise_transform == '':
        trans = Compose([ToTensor()])
    else:
        trans = Compose([ToTensor(), get_noise(noise_transform, percent_noise)])

    # Define download path
    data_path = 'data\\{}\\test\\'.format(data_name) if train else 'data\\{}\\train\\'.format(data_name)

    # Read data
    if data_name == 'MNIST':
        data_set = MNIST(data_path, train=train, download=True, transform=trans)
    else:
        raise NotImplementedError('Unkown data set: {}. '.format(data_name))

    return data_set