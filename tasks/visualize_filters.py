import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

from models import ConvClassificationModel, NonConvClassificationModel

def get_divisors(n):
    divisors = []
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            divisors.append((i, n//i))
    return divisors
    
def plot_filter_layer(filter_layer, conv_layer):
    num_output_channels = filter_layer.shape[0]
    num_input_channels = filter_layer.shape[1]

    for n in range(num_input_channels):
        divisors = get_divisors(num_output_channels)
        div_dist = [np.abs(divisors[i][0] - divisors[i][1]) for i in range(len(divisors))]
        min_index = div_dist.index(np.min(div_dist))
        num_rows, num_cols = divisors[min_index]

        fig = plt.figure(figsize=(5, 5)) 
        fig.suptitle('Convolutional Filters: Layer {}, Channel: {}'.format(conv_layer, n), fontsize=16)
        gs = gridspec.GridSpec(num_rows, num_cols)

        for i in range(num_rows):
            for j in range(num_cols):
                ax = plt.subplot(gs[i,j])
                ax.imshow(filter_layer[i*num_rows+j,n,:,:], cmap='gray')
                ax.axis('off')

    plt.show()

def visualize_filters():
    # Define transform
    trans = Compose([ToTensor()])

    # Download MNIST data set
    train_set = MNIST('data\\MNIST\\train\\', train=True, download=True, transform=trans)

    # MNIST digit dataset values
    input_size = np.prod(train_set.data.shape[1:])
    output_size = len(train_set.classes)

    # Create models
    nonconv_net = NonConvClassificationModel(input_size, output_size)
    conv_net = ConvClassificationModel()

    # Load models
    nonconv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_nonconv.model'))
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Plot non-convolutional weights
    #TODO

    # Plot convolutional filters
    filters1 = conv_net.conv1.weight.detach().numpy()
    filters2 = conv_net.conv2.weight.detach().numpy()
    plot_filter_layer(filters1, 1)
    plot_filter_layer(filters2, 2)