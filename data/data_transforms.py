import torch
import numpy as np

class AddNoise(object):
    def __init__(self, percent_noise: float):
        self.p_noise = percent_noise

    def _get_unique_indicies(self, tensor_dim):
        # Get indices to alter
        num_points = np.prod(tensor_dim)
        num_sample_points = int(num_points*self.p_noise)

        # Create mask
        index_samples = []
        while len(index_samples) < num_sample_points:
            # Sample indicies
            multi_index = []
            for dim in tensor_dim:
                multi_index.append(np.random.choice(dim))

            if not multi_index in index_samples:
                index_samples.append(multi_index)

        return index_samples


class AddGaussianNoise(AddNoise):
    def __init__(self, percent_noise: float, mean: float = 0.0, std: float = 1.0):
        super(AddGaussianNoise, self).__init__(percent_noise)
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Sample gaussian
        gaussian_samples = torch.randn(tensor.size()) * self.std + self.mean

        # Get indices to alter
        index_samples = self._get_unique_indicies(tensor.size())

        # Create mask
        mask = torch.zeros_like(tensor)
        for multi_index in index_samples:
            mask[tuple(multi_index)] = 1.0
        
        # Create gaussian mask
        gaussian_mask = torch.mul(mask, gaussian_samples)

        return tensor + gaussian_mask
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}) with p={2}'.format(self.mean, self.std, self.p_noise)


class AddSaltAndPepperNoise(AddNoise):
    def __call__(self, tensor):
        # Get indices to alter
        index_samples = self._get_unique_indicies(tensor.size())

        # Alter tensor
        for multi_index in index_samples:
            if np.random.choice(2) == 0:
                tensor[tuple(multi_index)] = 0.05
            else:
                tensor[tuple(multi_index)] = 0.95

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + ' with p={0}'.format(self.p_noise)


def get_noise(noise_name: str, percent_noise: float):
    if noise_name == 'gaussian':
        return AddGaussianNoise(percent_noise)
    elif noise_name == 'snp':
        return AddSaltAndPepperNoise(percent_noise)
    else:
        raise NotImplementedError('Unknown noise type: {0}. Available noise types: [gaussian, snp]'.format(noise_name))