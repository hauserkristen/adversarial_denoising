import os
import torch
import numpy as np
from bisect import bisect
import plotly.graph_objects as go
import plotly.offline as plt_offline

from .common import load_model_and_data, format_torch
from .DAG import DAG
from denoising import psnr

def create_histogram(noise_desc, noisy_psnr, adv_psnr, num_bins = 25):
    # Create ratio
    # Anything >1 would mean that the denoised adversarial example resulted in a more noisy image
    # Anything <1 would mean that the denoised adversarial example resulted in a less noisy image
    psnr_ratios = noisy_psnr / adv_psnr

    # Create figure
    fig = go.Figure(
        data=[
            go.Histogram(
                x=psnr_ratios,
                nbinsx=num_bins
            )
        ]
    )

    fig.update_layout(bargap=0.15)

    # Create directory if required
    if not os.path.exists('images//psnr'):
        os.mkdir('images//psnr')

    # Replace illegal characters
    filename = 'images//psnr//{}.html'.format(noise_desc)
    filename = filename.replace('.', '')

    # Save figure
    plt_offline.plot(fig, filename=filename, auto_open=False)

def calculate_psnr():
    # Set noise type
    noise_type = 'impulse'
    noise_param = 0.8

    # Load data
    net, test_set_original, test_set_noisy = load_model_and_data(noise_type, noise_param)

    # Test
    num_examples = len(test_set_noisy)
    noisy_psnr = np.zeros((num_examples))
    adv_psnr = np.zeros((num_examples))
    for i in range(num_examples):
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

        # Calculate PSNR for both denoised and adversarial denoised
        noisy_psnr[i] = psnr(denoised_result, orig_data).item()
        adv_psnr[i] = psnr(adv_denoised_result, orig_data).item()

    # Create graph
    noise_desc = '{}_{}'.format(noise_type, noise_param)
    create_histogram(noise_desc, noisy_psnr, adv_psnr)
