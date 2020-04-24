import os
import numpy as np
import matplotlib.pyplot as plt

img_dir = '../../noise2noise-pytorch/cifar/test'
f = np.random.choice(os.listdir(img_dir))

img = np.uint8(plt.imread(os.path.join(img_dir, f)) * 255)

fig, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].imshow(img)
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_title('Original')

noise_param = 500

for i in range(1, 4):
    dispersion = 50
    photons_per_pixel = np.random.negative_binomial(noise_param / dispersion,
                                                    1 / noise_param) / noise_param * dispersion
    print('Photons:', photons_per_pixel)
    noisy_img = np.random.poisson(np.array(img) / 255.0 * photons_per_pixel) / photons_per_pixel * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    axs[i].imshow(noisy_img)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title('%d photons per pixel' % int(photons_per_pixel))

plt.show()