import torch
import numpy as np
import plotly.graph_objects as go

from models import ConvClassificationModel
from data import get_data
from adv_attacks import FGSM, OnePixel
from .common import set_seed

def evaluate_fgsm_attack(net, test_data, loss_func, seed_val, rerun=False):
    epsilon = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies = []
    if rerun:
        for e in epsilon:
            set_seed(seed_val)
            attack = FGSM(e)
            acc = attack.run(net, test_data, loss_func)
            accuracies.append(acc)
    else:
        if isinstance(net, ConvClassificationModel):
            accuracies = [0.8887, 0.6611, 0.3618, 0.1411, 0.0489, 0.0202]
        else:
            accuracies = [0.7258, 0.2834, 0.0672, 0.0109, 0.0014, 0.0]

    return epsilon, accuracies

def evaluate_onepixel_attack(net, test_data, loss_func, seed_val, rerun=False):
    num_pixels = [1, 3, 5, 10, 20, 40]
    accuracies = []
    if rerun:
        for n in num_pixels:
            set_seed(seed_val)
            attack = OnePixel(100, 100, n)
            acc = attack.run(net, test_data, loss_func)
            accuracies.append(acc)
    else:
        # TODO: Need first value for both and last value for conv
        if isinstance(net, ConvClassificationModel):
            accuracies = [0.0, 0.9518, 0.9449, 0.9323, 0.9043, 0.0]
        else:
            accuracies = [0.0, 0.8965, 0.8871, 0.8653, 0.8274, 0.7543]

    return num_pixels, accuracies   

def plot_accuracy(data, title, axis_label):
    fig = go.Figure()
    
    fig.update_layout(
        title=title,
        xaxis_title=axis_label,
        yaxis_title='Accuracy (%)'
    )

    x_min = float('inf')
    x_max = -float('inf')
    for i, net_data in enumerate(data):
        if np.min(data[0]) < x_min:
            x_min = np.min(net_data[0])
        if np.max(data[0]) > x_max:
            x_max = np.max(net_data[0])

        fig.add_trace(
            go.Scatter(
                x=net_data[0],
                y=net_data[1],
                mode='lines'
            )
        )

    x_min -= 0.05*x_max
    x_max += 0.05*x_max
    
    fig.update_layout(
        xaxis={
            'range': [x_min,x_max]
        },
        yaxis={
            'range': [-0.05,1.05]
        }
    )

    fig.show()

def visualize_adv_affects_accuracy():
     # Hyper parameters
    seed = 2
    batch_size = 1
    data_name = 'MNIST'
    attack_name = 'OnePixel'

    # Download MNIST data set
    test_set = get_data(data_name, False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)

    # Create model
    conv_net = ConvClassificationModel()

    # Load model
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Get base accuracy
    set_seed(seed)
    base_conv_acc = conv_net.eval_model(test_loader)

    # Evaluate noise
    loss_func = torch.nn.NLLLoss()
    if attack_name == 'FGSM':
        axis_title = 'Epsilon'
        conv_x, conv_acc = evaluate_fgsm_attack(conv_net, test_loader, loss_func, seed)
    elif attack_name == 'OnePixel':
        axis_title = 'Number of Pixels'
        conv_x, conv_acc = evaluate_onepixel_attack(conv_net, test_loader, loss_func, seed)
    else:
        raise NotImplementedError('Unknown attack type: {0}. Available attacks; [FGSM, OnePixel]'.format(attack_name))

    # Insert base accuracy
    conv_x.insert(0,0)
    conv_acc.insert(0, base_conv_acc)

    # Plot results
    data = [(conv_x, conv_acc)]
    plot_accuracy(data, attack_name, axis_title)