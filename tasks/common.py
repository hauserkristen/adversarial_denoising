import torch
import numpy as np
import torch.nn.functional as F

def set_seed(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

def get_plots(data, conv_net):
    results = {}
    trans_data = conv_net._transform_input(data)

    layer_1_labels = ['Channel: 1', 'Channel: 2', 'Channel: 3']

    # After first convolution
    after_conv1 = conv_net.conv1(trans_data)
    results[(1, 'Filter')] = (after_conv1.detach().numpy()[0,:,:,:], layer_1_labels)

    # After first acivation
    after_act1 = F.relu(F.max_pool2d(after_conv1, 2, 2))
    results[(1, 'Activation')] = (after_act1.detach().numpy()[0,:,:,:], layer_1_labels)

    layer_2_labels = ['Channel: 1', 'Channel: 2', 'Channel: 3', 'Channel: 4', 'Channel: 5']

    # After second convolution
    after_conv2 = conv_net.conv2(after_act1)
    results[(2, 'Filter')] = (after_conv2.detach().numpy()[0,:,:,:], layer_2_labels)

    # After second acivation
    after_act2 = F.relu(F.max_pool2d(after_conv2, 2, 2))
    results[(2, 'Activation')] = (after_act2.detach().numpy()[0,:,:,:], layer_2_labels)

    return results