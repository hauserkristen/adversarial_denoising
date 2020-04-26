import torch
import numpy as np
from bisect import bisect
import plotly.graph_objects as go

from .common import load_model_and_data, format_torch
from .DAG import DAG
from denoising import psnr

def create_histogram(noisy_psnr, adv_psnr, num_bins = 25):
    # Create ratio
    # Anything >1 would mean that the denoised adversarial example resulted in a more noisy image
    # Anything <1 would mean that the denoised adversarial example resulted in a less noisy image
    psnr_ratios = noisy_psnr / adv_psnr

    # Identify min and max values to create bins
    min_value = np.min(psnr_ratios)
    max_value = np.max(psnr_ratios)
    delta = (max_value - min_value) / num_bins

    # Create bins
    bin_edges = np.arange(min_value - delta, max_value + 2*delta, delta)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Create histogram of ratio distribution
    ratio_dist = np.zeros_like(bins)
    for i in range(psnr_ratios.shape[0]):
        ratio_val = psnr_ratios[i]

        ratio_bin = bisect(bin_edges[1:], ratio_val)
        if ratio_bin >= bins.shape[0]:
            ratio_bin = bins.shape[0]-1

        ratio_dist[ratio_bin] += 1

    # Create figure
    fig = go.Figure(
        data=[
            go.Histogram(
                x=bins,
                y=ratio_dist
            )
        ]
    )
    fig.show()

def calculate_psnr():
    # Set noise type
    noise_type = 'gaussian'

    # Load data
    net, test_set_original, test_set_noisy = load_model_and_data(noise_type)

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
    create_histogram(noisy_psnr, adv_psnr)
    input('Press enter to close...')
