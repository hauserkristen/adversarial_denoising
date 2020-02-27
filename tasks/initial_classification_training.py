import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

from models import ConvClassificationModel, NonConvClassificationModel

def set_seed(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

def run_training(network: nn.Module, lr: float, m: float, n_epochs: int, train_data, test_data):
    # Define loss and optimizer
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=m)
    loss_func = nn.NLLLoss()

    for i in range(n_epochs):
        network.train_model(train_data, loss_func, optimizer)
        acc = network.eval_model(test_data)
        print('Epoch Accuracy: {}'.format(acc))
        
    return network

def train_mnist_digit_models():
    # Hyper parameters
    seed = 2
    num_epochs = 5
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 64

    # Define transform
    trans = Compose([ToTensor()])

    # Download MNIST data set
    train_set = MNIST('data\\MNIST\\train\\', train=True, download=True, transform=trans)
    test_set = MNIST('data\\MNIST\\test\\', train=False, download=True, transform=trans)

    # MNIST digit dataset values
    input_size = np.prod(train_set.data.shape[1:])
    output_size = len(train_set.classes)

    # Use torch data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)

    # Train and save models
    set_seed(seed)
    print('Training Non-Convolutional Classification Network...')
    nonconv_net = NonConvClassificationModel(input_size, output_size)
    nonconv_net = run_training(nonconv_net, learning_rate, momentum, num_epochs, train_loader, test_loader)
    torch.save(nonconv_net.save(), 'models\\pre_trained_models\\mnist_digit_nonconv.model')

    set_seed(seed)
    print('Training Convolutional Classification Network...')
    conv_net = ConvClassificationModel()
    conv_net = run_training(conv_net, learning_rate, momentum, num_epochs, train_loader, test_loader)
    torch.save(conv_net.save(), 'models\\pre_trained_models\\mnist_digit_conv.model')