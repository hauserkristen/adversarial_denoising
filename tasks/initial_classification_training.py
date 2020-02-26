import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models import ConvClassificationModel, NonConvClassificationModel

def set_seed(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

def run_training(network: nn.Module, lr: float, m: float, n_epochs: int, train_data, test_data):
    # Define loss and optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=m)
    loss_func = nn.NLLLoss()

    for i in range(num_epochs):
        net = network.train(train_data, loss_func, optimizer)
        acc = network.eval(test_data)
        print('Epoch Accuracy: {}'.format(acc))
        
    return net

# MNIST digit dataset values
input_size = 784
output_size = 10

# Hyper parameters
seed = 2
num_epochs = 5
learning_rate = 0.01
momentum = 0.5
            
# Train and save models
set_seed(seed)
nonconv_net = NonConvClassificationModel(input_size, output_size)
nonconv_net = run_training(nonconv_net, learning_rate, momentum, num_epochs, train_loader, test_loader)

set_seed(seed)
conv_net = ConvClassificationModel()
conv_net = run_training(conv_net, learning_rate, momentum, num_epochs, train_loader, test_loader)