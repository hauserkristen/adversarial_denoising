import os
import torch
import numpy as np
from bisect import bisect
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as plt_offline

from .common import load_model_and_data, format_torch
from .DAG import DAG

def display_images(noise_type: str, image_index: int, original_image: torch.Tensor, noisy_image: torch.Tensor, denoised_image: torch.Tensor, adv_image: torch.Tensor, denoised_adv_image: torch.Tensor):
    orig_np = original_image.detach().numpy().squeeze(0)
    noisy_np = noisy_image.detach().numpy().squeeze(0)
    denoised_np = denoised_image.detach().numpy().squeeze(0)
    adv_image_np = adv_image.detach().numpy().squeeze(0)
    adv_denoised_np = denoised_adv_image.detach().numpy().squeeze(0)

    # Flip axes back
    orig_np = np.moveaxis(orig_np, 0, -1)
    noisy_np = np.moveaxis(noisy_np, 0, -1)
    denoised_np = np.moveaxis(denoised_np, 0, -1)
    adv_image_np = np.moveaxis(adv_image_np, 0, -1)
    adv_denoised_np = np.moveaxis(adv_denoised_np, 0, -1)

    # Calculate difference
    noise_np = noisy_np - orig_np
    adv_noise_np = adv_image_np - orig_np

    fig = make_subplots(
        rows=2, 
        cols=4,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=['Clean Image', 'Noise', 'Noisy Image', 'Denoised Image', 'Noise Distribution', 'Adversarial Noise', 'Adversarial Noisy Image', 'Denoised Adversarial Image'])

    plots = [
        orig_np, noise_np, noisy_np, denoised_np,
        None, adv_noise_np, adv_image_np, adv_denoised_np
    ]

    for i, p in enumerate(plots):
        row = (i // 4) + 1
        col = (i % 4) + 1

        if p is None:
            # Create histogram of noise distribution
            noise_vals = []
            adversarial_vals = []
            for j in range(noise_np.shape[0]):
                for k in range(noise_np.shape[1]):
                    for l in range(noise_np.shape[2]):
                        noise_vals.append(noise_np[j,k,l])
                        adversarial_vals.append(adv_noise_np[j,k,l])

            # Add traces
            fig.add_trace(
                go.Histogram(
                    x=noise_vals,
                    nbinsx=25,
                    name='Noise'
                ),
                row=row,
                col=col
            )
            fig.add_trace(
                go.Histogram(
                    x=adversarial_vals,
                    nbinsx=25,
                    name='Adversarial'
                ),
                row=row,
                col=col
            )
            fig.update_layout(bargap=0.15)

        else:
            # Define range
            zmax = [1,1,1,1]
            zmin = [0,0,0,0]
            if col == 2:
                zmin = [-1,-1,-1,-1]

            fig.add_trace(
                go.Image(
                    z=p,
                    colormodel='rgb',
                    zmax=zmax,
                    zmin=zmin
                ),
                row=row,
                col=col
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)

    # Format figure
    fig.update_layout(
        legend={
            'x': -0.1, 
            'y': 0.0
        }
    )

    # Save figure
    if not os.path.exists('images//{}'.format(noise_type)):
        os.mkdir('images//{}'.format(noise_type))

    filename = 'images//{}//image_{}.html'.format(noise_type, image_index)
    plt_offline.plot(fig, filename=filename, auto_open=False)

def visualize_examples():
    # Set noise type and number of samples to save
    noise_type = 'poisson'
    num_samples = 25

    # Load data
    net, test_set_original, test_set_noisy = load_model_and_data(noise_type)

    # Randomly choose indices
    num_examples = len(test_set_noisy)
    save_indices = np.random.randint(num_examples, size=num_samples)

    # Test
    for i in save_indices:
        orig_data, orig_label = test_set_original[i]
        noisy_data, _ = test_set_noisy[i]

        # Proper format
        orig_data = orig_data.unsqueeze(0).float()
        noisy_data = noisy_data.unsqueeze(0).float()

        # Denoise
        denoised_result = net(noisy_data)
        denoised_result = format_torch(denoised_result)

        # Call attack
        adversarial_noise = DAG(net, orig_data, noisy_data)
        adversarial_data = orig_data + adversarial_noise
        adversarial_data = format_torch(adversarial_data)

        # Denoise
        adv_denoised_result = net(adversarial_data)
        adv_denoised_result = format_torch(adv_denoised_result)

        display_images(noise_type, i, orig_data, noisy_data, denoised_result, adversarial_data, adv_denoised_result)
