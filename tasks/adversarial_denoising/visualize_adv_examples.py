import torch
import numpy as np
from bisect import bisect
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import load_model_and_data, format_torch
from .DAG import DAG

def display_images(original_image: torch.Tensor, noisy_image: torch.Tensor, denoised_image: torch.Tensor, adv_image: torch.Tensor, denoised_adv_image: torch.Tensor):
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

    # Calculate difference before casting
    noise_np = noisy_np - orig_np
    adv_noise_np = adv_image_np - orig_np

    # Enforce integers
    orig_np = orig_np.astype(np.uint8)
    noisy_np = noisy_np.astype(np.uint8)
    denoised_np = denoised_np.astype(np.uint8)
    adv_image_np = adv_image_np.astype(np.uint8)
    adv_denoised_np = adv_denoised_np.astype(np.uint8)

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
            # Create bins
            bin_edges = np.arange(-255,260,5)
            bins = (bin_edges[1:] + bin_edges[:-1]) / 2

            # Create histogram of noise distribution
            guassian_dist = np.zeros_like(bins)
            adversarial_dist = np.zeros_like(bins)
            for j in range(noise_np.shape[0]):
                for k in range(noise_np.shape[1]):
                    for l in range(noise_np.shape[2]):
                        gausian_val = noise_np[j,k,l]
                        adv_val = adv_noise_np[j,k,l]

                        gaussian_bin = bisect(bin_edges[1:], gausian_val)
                        if gaussian_bin >= bins.shape[0]:
                            gaussian_bin = bins.shape[0]-1
                        adv_bin = bisect(bin_edges[1:], adv_val)
                        if adv_bin >= bins.shape[0]:
                            adv_bin = bins.shape[0]-1

                        guassian_dist[gaussian_bin] += 1
                        adversarial_dist[adv_bin] += 1

            # Add traces
            fig.add_trace(
                go.Bar(
                    x=bins,
                    y=guassian_dist,
                    name='Gaussian'
                ),
                row=row,
                col=col
            )
            fig.add_trace(
                go.Bar(
                    x=bins,
                    y=adversarial_dist,
                    name='Adversarial'
                ),
                row=row,
                col=col
            )
        else:
            # Define range
            zmax = [255,255,255,255]
            zmin = [0,0,0,0]
            if col == 2:
                zmin = [-255,-255,-255,-255]

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

    fig.show()

def visualize_examples():
    # Set noise type
    noise_type = 'gaussian'

    # Load data
    net, test_set_original, test_set_noisy = load_model_and_data(noise_type)

    # Test
    for i in range(len(test_set_noisy)):
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

        display_images(orig_data, noisy_data, denoised_result, adversarial_data, adv_denoised_result)
        input('Press enter to continue...')
