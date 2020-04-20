import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import get_data
from denoising import UNet
from tasks import set_seed

def display_images(original_image: torch.Tensor, noisy_image: torch.Tensor, denoised_image: torch.Tensor):
    orig_np = original_image.detach().numpy()
    noisy_np = noisy_image.detach().numpy().squeeze(0)
    denoised_np = denoised_image.detach().numpy().squeeze(0)

    # Flip axes back
    orig_np = np.moveaxis(orig_np, 0, -1)
    noisy_np = np.moveaxis(noisy_np, 0, -1)
    denoised_np = np.moveaxis(denoised_np, 0, -1)

    # Enforce integers
    orig_np = orig_np.astype(np.uint8)
    noisy_np = noisy_np.astype(np.uint8)
    denoised_np = denoised_np.astype(np.uint8)

    fig = make_subplots(
        rows=1, 
        cols=3,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=['Clean Image', 'Noisy Image', 'Denoised Image'])

    fig.add_trace(
        go.Image(
            z=orig_np
        ),
        row=1,
        col=1
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=1)

    fig.add_trace(
        go.Image(
            z=noisy_np
        ),
        row=1,
        col=2
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=2)

    fig.add_trace(
        go.Image(
            z=denoised_np
        ),
        row=1,
        col=3
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=3)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=1, col=3)

    fig.show()

def load_model():
    # Path to pretrained network weights
    filename = 'denoising\\models\\n2n_gaussian_std_50.pt'

    # Load network
    net = UNet()
    net.load_state_dict(torch.load(filename))
    net.eval()

    # Parameters
    seed = 2
    batch_size = 1

    set_seed(seed)

    # Load CIFAR-10 data
    test_set_original = get_data('CIFAR10', False)
    test_set_noisy = get_data('CIFAR10', False, 'gaussian')

    # Test
    with torch.no_grad():
        for i in range(len(test_set_noisy)):
            orig_data, orig_label = test_set_original[i]
            noisy_data, _ = test_set_noisy[i]
            noisy_data = noisy_data.unsqueeze(0).float()
            denoised_result = net(noisy_data)

            display_images(orig_data, noisy_data, denoised_result)
            input()
