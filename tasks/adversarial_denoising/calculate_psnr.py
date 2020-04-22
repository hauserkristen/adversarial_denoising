import torch
import numpy as np

from .common import load_model_and_data, format_torch
from .DAG import DAG
from denoising import psnr

def calculate_psnr():
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

        # Calculate PSNR for both denoised and adversarial denoised
        denoised_psnr = psnr(denoised_result, orig_data)
        adv_denoised_psnr = psnr(adv_denoised_result, orig_data)
