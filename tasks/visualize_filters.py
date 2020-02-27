import torch
import numpy as np
import matplotlib.pyplot as plt

from models import ConvClassificationModel, NonConvClassificationModel
from data import get_data

def plot_filter_layer(filter_layers):
    # Visualize each filter
    for filter_index, filter_layer in enumerate(filter_layers):
        num_output_channels = filter_layer.shape[0]
        num_input_channels = filter_layer.shape[1]

        # Create figure
        fig = plt.figure(figsize=(10,10)) 

        for o in range(num_output_channels):
            for i in range(num_input_channels):
                # Get filter
                vis_filter = filter_layer[o,i,:,:]
                # Plot
                ax = plt.subplot2grid((vis_filter.shape[0], vis_filter.shape[1]), (o,i))
                ax.imshow(vis_filter, cmap='gray')
                ax.axis('off')
                ax.set_title('Conv Layer: {}\nI/O Channel: {}/{}'.format(filter_index, i, o))
        
        plt.tight_layout()
        plt.show()

def visualize_filters():
    # Download MNIST data set
    train_set = get_data('MNIST', True)

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
    filters = [
        conv_net.conv1.weight.detach().numpy(),
        conv_net.conv2.weight.detach().numpy()
    ]
    plot_filter_layer(filters)