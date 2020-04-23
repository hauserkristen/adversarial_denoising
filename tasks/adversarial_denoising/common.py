import torch

from data import get_data
from denoising import UNet
from tasks import set_seed

def format_torch(x):
    x = x.clamp(0,1)
    return x

def load_model_and_data(noise_type: str, seed: int = 2):
    # Path to pretrained network weights
    if noise_type == 'gaussian':
        filename = 'denoising\\models\\n2n_gaussian_std_50.pt'
    else:
        filename = 'denoising\\models\\n2n_poisson_lambda_150.pt'

    # Load network
    net = UNet()
    net.load_state_dict(torch.load(filename))
    net.eval()
    
    # Fix network
    for p in net.parameters():
        p.requires_grad = False

    # Set random seed
    set_seed(seed)

    # Load CIFAR-10 data
    test_set_original = get_data('CIFAR10', False)
    test_set_noisy = get_data('CIFAR10', False, noise_type)

    return net, test_set_original, test_set_noisy