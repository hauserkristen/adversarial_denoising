import torch
import numpy as np
import plotly.graph_objects as go

from models import ConvClassificationModel
from data import get_data
from tasks import set_seed

def evaluate_noise(net, data_name, noise_name, b_size, seed_val, rerun=False):
    percent_noise = [0.1, 0.2, 0.3, 0.4, 0.5]
    accuracies = []
    if rerun:
        for p in percent_noise:
            set_seed(seed_val)
            test_data = get_data(data_name, False, noise_name, p)
            test_data_loader = torch.utils.data.DataLoader(test_data, b_size, shuffle=True)

            set_seed(seed_val)
            acc = net.eval_model(test_data_loader)
            accuracies.append(acc)
        print(percent_noise, accuracies)
    elif noise_name == 'snp':
        if isinstance(net, ConvClassificationModel):
            accuracies = [0.9592, 0.9065, 0.7882, 0.65, 0.5002]
        else:
            accuracies = [0.9171, 0.8294, 0.6708, 0.5282, 0.4156]
    else:
        if isinstance(net, ConvClassificationModel):
            accuracies = [0.9199, 0.858, 0.7056, 0.5639, 0.444]
        else:
            accuracies = [0.9639, 0.933, 0.846, 0.7141, 0.5842]

    return percent_noise, accuracies
        
def plot_accuracy(data, labels):
    fig = go.Figure()
    
    fig.update_layout(
        xaxis_title='% Image Replaced with Noise',
        yaxis_title='Accuracy (%)'
    )

    for i, net_data in enumerate(data):
        fig.add_trace(
            go.Scatter(
                x=net_data[0],
                y=net_data[1],
                mode='lines',
                name=labels[i]
            )
        )

    fig.update_layout(
        xaxis={
            'range': [-0.05,0.55]
        },
        yaxis={
            'range': [-0.05,1.05]
        },
        legend={
            'x': 0.005,
            'y': 0.01
        }
    )

    fig.show()

def visualize_noisy_affects_accuracy():
    # Hyper parameters
    seed = 2
    batch_size = 64
    data_name = 'MNIST'

    # Download MNIST data set
    test_set = get_data(data_name, False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)

    conv_net = ConvClassificationModel()

    # Load model
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Get base accuracy
    set_seed(seed)
    base_conv_acc = conv_net.eval_model(test_loader)

    data = []
    for noise_type in ['snp', 'gaussian_gray']:
        # Evaluate noise
        conv_percent, conv_acc = evaluate_noise(conv_net, 'MNIST', noise_type, batch_size, seed)

        # Insert base accuracy
        conv_percent.insert(0,0)
        conv_acc.insert(0, base_conv_acc)

        # Plot results
        data.append((conv_percent, conv_acc))

    plot_accuracy(data, ['Salt-And-Pepper Noise', 'Gaussian Noise'])