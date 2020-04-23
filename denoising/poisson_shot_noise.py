import os
import numpy as np
import matplotlib.pyplot as plt

img_dir = '../../noise2noise-pytorch/cifar/test'
f = np.random.choice(os.listdir(img_dir))

img = np.uint8(plt.imread(os.path.join(img_dir, f)) * 255)

print(img)

fig, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].imshow(img)
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_title('Original')

photon_counts = [5000, 2500, 1000]
for i in range(1, 4):
    noisy_img = np.random.poisson(np.array(img) / 255.0 * photon_counts[i - 1]) / photon_counts[i - 1] * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    axs[i].imshow(noisy_img)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title('%d photons per pixel' % photon_counts[i - 1])

plt.show()

fig, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].imshow(img)
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_title('Original')

for i in range(1, 4):
    photons_per_pixel = np.random.uniform(100,
                                          1024)  # 1024 chosen rather arbitrarily based on 32x32 images
    noisy_img = np.random.poisson(np.array(img) / 255.0 * photons_per_pixel) / photons_per_pixel * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    axs[i].imshow(noisy_img)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title('%d photons per pixel' % photon_counts[i - 1])

plt.show()