import torch
import numpy as np

# Noise types for RGB CIFAR10 data set

class ToRGBTensor(object):
    """
    Default ToTensor transform scale input to [0,1], this does not
    """
    def __call__(self, x):
        # Convert from PIL image to numpy
        result = np.asarray(x)

        # Move axis to enforce same style as previous input
        result = np.moveaxis(result, -1, 0)

        # Convert from numpy to torch tensor
        result = torch.from_numpy(result)
        return result

class ScaleImage(object):
    def __call__(self, x):
        # Convert to float
        result = x.float()

        # Scale from [0,255] to [0,1]
        result = torch.div(result, 255)
        return result

class AddGaussianNoise(object):
    def __init__(self, mean: float = 0.0, max_std: float = 50.0):
        self.max_std = max_std
        self.mean = mean
        
    def __call__(self, tensor):
        # Sample std dev
        std = np.random.uniform(0, self.max_std)

        # Sample gaussian
        gaussian_samples = torch.randn(tensor.size()) * std + self.mean

        # Create mask
        result = tensor.clone().float()
        result += gaussian_samples
        
        # Clamp
        result[result < 0] = 0
        result[result > 255] = 255

        return result.type(torch.uint8)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}) with p={2}'.format(self.mean, self.std, self.p_noise)

# Noise types for the grayscale MNIST dataset
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

class AddGaussianGrayScaleNoise(AddNoise):
    def __init__(self, percent_noise: float, mean: float = 0.5, std: float = 0.2):
        super(AddGaussianGrayScaleNoise, self).__init__(percent_noise)
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Sample gaussian
        gaussian_samples = torch.randn(tensor.size()) * self.std + self.mean

        # Get indices to alter
        index_samples = self._get_unique_indicies(tensor.size())

        # Create mask
        result = tensor.clone()
        for multi_index in index_samples:
            result[tuple(multi_index)] = gaussian_samples[tuple(multi_index)]
        
        # Clamp
        result[result < 0.0] = 0.0
        result[result > 1.0] = 1.0

        return result
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}) with p={2}'.format(self.mean, self.std, self.p_noise)

class AddSaltAndPepperNoise(AddNoise):
    def __call__(self, tensor):
        # Get indices to alter
        index_samples = self._get_unique_indicies(tensor.size())

        # Alter tensor
        result = tensor.clone()
        for multi_index in index_samples:
            if np.random.choice(2) == 0:
                result[tuple(multi_index)] = 0.05
            else:
                result[tuple(multi_index)] = 0.95

        return result

    def __repr__(self):
        return self.__class__.__name__ + ' with p={0}'.format(self.p_noise)


def get_noise(noise_name: str, percent_noise: float):
    if noise_name == 'gaussian_gray':
        return AddGaussianGrayScaleNoise(percent_noise)
    elif noise_name == 'gaussian':
        return AddGaussianNoise()
    elif noise_name == 'snp':
        return AddSaltAndPepperNoise(percent_noise)
    else:
        raise NotImplementedError('Unknown noise type: {0}. Available noise types: [gaussian_gray, gaussian, snp]'.format(noise_name))