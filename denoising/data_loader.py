import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from PIL import Image


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    noise = (params.noise_type, params.noise_param)

    dataset = NoisyDataset(root_dir, redux, params.crop_size,
                           clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    if single:  # use if testing to only load one image
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')

    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):

    def __init__(self, root_dir, redux, crop_size, clean_targets=False, noise_dist=('gaussian', 50.), seed=None):
        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            # Add noise and clip
            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)

    def _corrupt(self, img):
        """Corrupts images (Gaussian or Poisson)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img = Image.open(img_path).convert('RGB')

        # Corrupt source image
        tmp = self._corrupt(img)
        source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        return source, target