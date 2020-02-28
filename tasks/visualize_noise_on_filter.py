import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models import ConvClassificationModel, NonConvClassificationModel
from data import get_data


def set_seed(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

def plot_result(layer_index: int, layer_type: str, data: np.ndarray, noisy_data: np.ndarray = None):
    # Create figure
    num_rows = data.shape[0]
    num_cols = 1 if noisy_data is None else 2
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10,10)) 

    for i in range(num_rows):
        if noisy_data is None:
            # Get result
            vis_data = data[i,:,:]

            # Plot
            ax[i].imshow(vis_data, cmap='gray')
            ax[i].axis('off')
            ax[i].set_title('{} at Layer: {}\nChannel: {}'.format(layer_type, layer_index, i))
        else:
            # Get result
            vis_data = data[i,:,:]
            vis_data_n = noisy_data[i,:,:]

            # Plot
            ax[i, 0].imshow(vis_data, cmap='gray')
            ax[i, 0].axis('off')
            ax[i, 0].set_title('{} at Layer: {}\nChannel: {}'.format(layer_type, layer_index, i))
            ax[i, 1].imshow(vis_data_n, cmap='gray')
            ax[i, 1].axis('off')
            ax[i, 1].set_title('Noisy {} at Layer: {}\nChannel: {}'.format(layer_type, layer_index, i))
    
    plt.tight_layout()
    plt.show()

def plot_after_filters(conv_net: torch.Tensor, data, noisy_data: torch.Tensor = None):
    trans_data = conv_net._transform_input(data)
    if not noisy_data is None:
        trans_data_n = conv_net._transform_input(noisy_data)

    # After first convolution
    after_conv1 = conv_net.conv1(trans_data)
    if not noisy_data is None:
        after_conv1_n = conv_net.conv1(trans_data_n)
        plot_result(1, 'Convolution', after_conv1.detach().numpy()[0,:,:,:], after_conv1_n.detach().numpy()[0,:,:,:])
    else:
        plot_result(1, 'Convolution',  after_conv1.detach().numpy()[0,:,:,:])

    # After first acivation
    after_act1 = F.relu(F.max_pool2d(after_conv1, 2, 2))
    if not noisy_data is None:
        after_act1_n = F.relu(F.max_pool2d(after_conv1_n, 2, 2))
        plot_result(1, 'Activation', after_act1.detach().numpy()[0,:,:,:], after_act1_n.detach().numpy()[0,:,:,:])
    else:
        plot_result(1, 'Activation', after_act1.detach().numpy()[0,:,:,:])

    # After second convolution
    after_conv2 = conv_net.conv2(after_act1)
    if not noisy_data is None:
        after_conv2_n = conv_net.conv2(after_act1_n)
        plot_result(2, 'Convolution', after_conv2.detach().numpy()[0,:,:,:], after_conv2_n.detach().numpy()[0,:,:,:])
    else:
        plot_result(2, 'Convolution', after_conv2.detach().numpy()[0,:,:,:])

    # After second acivation
    after_act2 = F.relu(F.max_pool2d(after_conv2, 2, 2))
    if not noisy_data is None:
        after_act2_n = F.relu(F.max_pool2d(after_conv2_n, 2, 2))
        plot_result(2, 'Activation', after_act2.detach().numpy()[0,:,:,:], after_act2_n.detach().numpy()[0,:,:,:])
    else:
        plot_result(2, 'Activation', after_act2.detach().numpy()[0,:,:,:])
    

def visualize_noisy_affects_filter():
    # Hyper parameters
    seed = 2
    data_name = 'MNIST'

    # Download MNIST data set
    set_seed(seed)
    test_set = get_data(data_name, False)
    test_set_n = get_data(data_name, False, 'snp', 0.3)

    # MNIST digit dataset values
    input_size = np.prod(test_set.data.shape[1:])
    output_size = len(test_set.classes)

    # Create models
    conv_net = ConvClassificationModel()

    # Load models
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    for i, (clean_data, label) in enumerate(test_set):
        noisy_data, noisy_label = test_set_n[i]

        clean_data = clean_data.view(1,*clean_data.size()).float()
        noisy_data = noisy_data.view(1,*noisy_data.size()).float()
        plot_after_filters(conv_net, clean_data, noisy_data)