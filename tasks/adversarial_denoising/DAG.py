import torch
import numpy as np

def _check_condition(output, target):
    condition = torch.zeros_like(output)

    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            for k in range(output.shape[3]):
                output_val = output[0,i,j,k].type(torch.uint8)
                target_val = target[0,i,j,k].type(torch.uint8)

                if output_val == target_val:
                    condition[0,i,j,k] = 0.0
                else:
                    condition[0,i,j,k] = 1.0

    return condition

def DAG(model, clean_image, adv_target, num_iterations=20, gamma=0.07):
    # Creat adversarial image to be optimized
    adv_noise = torch.zeros_like(clean_image)
    adv_noise.requires_grad = True

    for a in range(num_iterations):
        # Add adversarial noise
        adv_image = clean_image + adv_noise
        adv_image = adv_image.clamp(0, 255)

        # Denoise
        denoised_image = model(adv_image)
        denoised_image = denoised_image.clamp(0, 255)

        # Check for condition
        condition = _check_condition(denoised_image, adv_target)

        if torch.sum(condition) == 0:
            print("Condition Reached, no target difference")
            break

        # Finding pixels to purturb
        adv_log = torch.mul(denoised_image, adv_target)

        # Getting the values of the original output
        clean_log = torch.mul(denoised_image, clean_image)

        # Finding r_m
        adv_direction = adv_log - clean_log
        r_m = torch.mul(adv_direction, condition)
        r_m.requires_grad_()

        # Summation
        r_m_sum = r_m.sum()
        r_m_sum.requires_grad_()

        # Finding gradient with respect to image
        r_m_grad = torch.autograd.grad(r_m_sum, adv_image, retain_graph=True)[0]
        
        # Calculating Magnitude of the gradient
        r_m_grad_mag = r_m_grad.norm()
        
        if r_m_grad_mag == 0:
            print("Condition Reached, no gradient")
            break

        #Calculating final value of r_m
        r_m_norm = gamma * r_m_grad

        # Updating the noise
        adv_noise = adv_noise + r_m_norm
        adv_noise = torch.clamp(adv_noise, -255, 255)

    return adv_noise