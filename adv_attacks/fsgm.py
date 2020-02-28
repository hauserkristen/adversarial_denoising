import torch

from .base import AdversarialAttack

class FGSM(AdversarialAttack):
    def __init__(self, eps: float):
        self.epsilon = eps

    def attack(self, network: torch.nn.Module, data: torch.Tensor, label: torch.Tensor, data_gradient: torch.Tensor):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_gradient.sign()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = data + self.epsilon * sign_data_grad

        # Adding clipping to maintain [0,1] range
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

        # Return the perturbed image
        return perturbed_data