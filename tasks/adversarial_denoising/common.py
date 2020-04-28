import torch

from data import get_data
from denoising import UNet
from tasks import set_seed

def format_torch(x):
    x = x.clamp(0,1)
    return x

def load_model_and_data(noise_type: str, noise_param: int, seed: int = 2):
    # Path to pretrained network weights
    if noise_type == 'gaussian' and noise_param == 50:
        filename = 'denoising\\models\\n2n_gaussian_std_50.pt'
    elif noise_type == 'gaussian' and noise_param == 100:
        filename = 'denoising\\models\\n2n_gaussian_std_100.pt'
    elif noise_type == 'poisson' and noise_param == 100:
        filename = 'denoising\\models\\n2n_poisson_100.pt'
    elif noise_type == 'poisson' and noise_param == 200:
        filename = 'denoising\\models\\n2n_poisson_200.pt'
    elif noise_type == 'poisson' and noise_param == 500:
        filename = 'denoising\\models\\n2n_poisson_500.pt'
    elif noise_type == 'impulse' and noise_param == 0.5:
        filename = 'denoising\\models\\n2n_impulse_std50_p0.5.pt'
    elif noise_type == 'impulse' and noise_param == 0.8:
        filename = 'denoising\\models\\n2n_impulse_std50_p0.8.pt'
    else:
        raise Exception('Unknown model parameterization.')

    # Load network
    net = UNet()
    net.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    net.eval()
    
    # Fix network
    for p in net.parameters():
        p.requires_grad = False

    # Set random seed
    set_seed(seed)

    # Load CIFAR-10 data
    test_set_original = get_data('CIFAR10', False)
    test_set_noisy = get_data('CIFAR10', False, noise_type, noise_param)

    return net, test_set_original, test_set_noisy