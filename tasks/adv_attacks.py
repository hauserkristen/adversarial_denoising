import torch
import numpy as np
import matplotlib.pyplot as plt

from models import ConvClassificationModel, NonConvClassificationModel
from data import get_data
from adv_attacks import FGSM, OnePixel

def set_seed(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)


def evaluate_fgsm_attack(net, test_data, loss_func, seed_val):
    epsilon = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies = []
    for e in epsilon:
        set_seed(seed_val)
        attack = FGSM(e)
        acc = attack.run(net, test_data, loss_func)
        accuracies.append(acc)

    return epsilon, accuracies

def evaluate_onepixel_attack(net, test_data, loss_func, seed_val):
    num_pixels = [1, 2, 3, 4, 5, 7, 10]
    accuracies = []
    for n in num_pixels:
        set_seed(seed_val)
        attack = OnePixel(100, 400, n)
        acc = attack.run(net, test_data, loss_func)
        accuracies.append(acc)

    return num_pixels, accuracies   

def plot_accuracy(data, labels, title, axis_label):
    fig = plt.figure()
    plt.xlabel(axis_label)
    plt.ylabel('Accuracy (%)')

    for i, net_data in enumerate(data):
        plt.plot(*net_data, label=labels[i]) 
        
    fig.suptitle(title)
    plt.legend()
    plt.show()


def visualize_adv_affects_accuracy():
     # Hyper parameters
    seed = 2
    batch_size = 1
    data_name = 'MNIST'
    attack_name = 'FGSM'

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
    loss_func = torch.nn.NLLLoss()
    if attack_name == 'FGSM':
        axis_title = 'Epsilon'
        nonconv_x, nonconv_acc = evaluate_fgsm_attack(nonconv_net, test_loader, loss_func, seed)
        conv_x, conv_acc = evaluate_fgsm_attack(conv_net, test_loader, loss_func, seed)
    elif attack_name == 'OnePixel':
        axis_title = 'Number of Pixels'
        nonconv_x, nonconv_acc = evaluate_onepixel_attack(nonconv_net, test_loader, loss_func, seed)
        conv_x, conv_acc = evaluate_onepixel_attack(conv_net, test_loader, loss_func, seed)
    else:
        raise NotImplementedError('Unknown attack type: {0}. Available attacks; [FGSM, OnePixel]'.format(attack_name))

    # Insert base accuracy
    nonconv_x.insert(0,0)
    nonconv_acc.insert(0, base_nonconv_acc)
    conv_x.insert(0,0)
    conv_acc.insert(0, base_conv_acc)

    # Plot results
    labels = ['Non-Convolutional Network', 'Convolutional Network']
    data = [(nonconv_x, nonconv_acc), (conv_x, conv_acc)]
    plot_accuracy(data, labels, attack_name, axis_title)