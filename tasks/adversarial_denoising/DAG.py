import torch
import numpy as np

from .common import format_torch

EPSILON = 0.04

def _check_condition(output, target):
    condition = torch.zeros_like(output)

    for i in range(output.shape[2]):
        for j in range(output.shape[3]):
            output_val = output[0,:,i,j]
            target_val = target[0,:,i,j]

            if torch.abs(output_val - target_val).sum().item() < EPSILON:
                condition[0,:,i,j] = 0.0
            else:
                condition[0,:,i,j] = 1.0
    return condition

def DAG(model, clean_image, adv_target, num_iterations=40, gamma=0.1):
    # Creat adversarial image to be optimized
    adv_noise = torch.zeros_like(clean_image)
    adv_noise.requires_grad = True

    for a in range(num_iterations):
        # Add adversarial noise
        adv_image = clean_image + adv_noise
        adv_image = format_torch(adv_image)

        # Denoise
        denoised_image = model(adv_image)
        denoised_image = format_torch(denoised_image)

        # Check for target pixels
        condition = _check_condition(denoised_image, adv_target)
        num_pixels_to_modify = np.count_nonzero(condition.detach().numpy() == 1.0)

        if num_pixels_to_modify == 0:
            print('Complete, No target difference')
            break

        # Finding pixels to purturb
        adv_log = torch.mul(denoised_image, adv_target)

        # Getting the values of the original output
        clean_log = torch.mul(denoised_image, clean_image)

        # Finding r_m
        adv_direction = adv_log - clean_log
        r_m = torch.mul(adv_direction, condition)
        r_m_sum = r_m.sum()

        # Finding gradient with respect to image
        r_m_grad = torch.autograd.grad(r_m_sum, adv_image, retain_graph=True)[0]
        
        # Calculating magnitude of the gradient
        r_m_grad_mag = r_m_grad.norm()
        
        if r_m_grad_mag == 0:
            print('Complete, no gradient to apply')
            break

        #Calculating final value of r_m
        r_m_norm = (gamma / r_m_grad_mag) * r_m_grad
        r_m_norm = torch.mul(r_m_norm, condition)

        # Updating the noise
        adv_noise = adv_noise + r_m_norm
        adv_noise = torch.clamp(adv_noise, -1, 1)

    return adv_noise