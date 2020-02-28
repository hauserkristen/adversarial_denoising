import torch
import numpy as np
import matplotlib.pyplot as plt

from models import ConvClassificationModel, NonConvClassificationModel
from data import get_data

def set_seed(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)


def evaluate_noise(net, data_name, noise_name, b_size, seed_val):
    percent_noise = [0.1, 0.2, 0.3, 0.4, 0.5]
    accuracies = []
    for p in percent_noise:
        set_seed(seed_val)
        test_data = get_data(data_name, False, noise_name, p)
        test_data_loader = torch.utils.data.DataLoader(test_data, b_size, shuffle=True)

        set_seed(seed_val)
        acc = net.eval_model(test_data_loader)
        accuracies.append(acc)

    return percent_noise, accuracies
        

def plot_accuracy(data, labels, title):
    fig = plt.figure()
    plt.xlabel('% Image Replaced with Noise')
    plt.ylabel('Accuracy (%)')

    for i, net_data in enumerate(data):
        plt.plot(*net_data, label=labels[i]) 
        
    fig.suptitle(title)
    plt.legend()
    plt.show()


def visualize_noisy_affects_accuracy():
    # Hyper parameters
    seed = 2
    batch_size = 64
    data_name = 'MNIST'
    noise_type = 'snp'

    # Download MNIST data set
    test_set = get_data(data_name, False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)

    # MNIST digit dataset values
    input_size = np.prod(test_set.data.shape[1:])
    output_size = len(test_set.classes)

    # Create models
    nonconv_net = NonConvClassificationModel(input_size, output_size)
    conv_net = ConvClassificationModel()

    # Load models
    nonconv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_nonconv.model'))
    conv_net.load(torch.load('models\\pre_trained_models\\mnist_digit_conv.model'))

    # Get base accuracy
    set_seed(seed)
    base_nonconv_acc = nonconv_net.eval_model(test_loader)
    set_seed(seed)
    base_conv_acc = conv_net.eval_model(test_loader)

    # Evaluate noise
    nonconv_percent, nonconv_acc = evaluate_noise(nonconv_net, 'MNIST', noise_type, batch_size, seed)
    conv_percent, conv_acc = evaluate_noise(conv_net, 'MNIST', noise_type, batch_size, seed)

    # Insert base accuracy
    nonconv_percent.insert(0,0)
    nonconv_acc.insert(0, base_nonconv_acc)
    conv_percent.insert(0,0)
    conv_acc.insert(0, base_conv_acc)

    # Plot results
    labels = ['Non-Convolutional Network', 'Convolutional Network']
    data = [(nonconv_percent, nonconv_acc), (conv_percent, conv_acc)]
    plot_accuracy(data, labels, 'Salt-And-Pepper Noise')