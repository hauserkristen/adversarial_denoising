import numpy as np
import matplotlib.pyplot as plt

dispersions = [2, 4, 8, 10, 20, 50, 75, 100]

noise_param = 500
poisson = np.random.poisson(noise_param, 1000000)
plt.hist(poisson, bins=50, label='poisson', alpha=0.5, density=True, histtype='step', linewidth=2)
for d in dispersions:
    photons_per_pixel = np.random.negative_binomial(noise_param / d, 1 / noise_param, 1000000) / noise_param * d
    plt.hist(photons_per_pixel, bins=50, label='nb dispersion=%d' % d, alpha=0.5, density=True, histtype='step', linewidth=2)

plt.legend(loc='best')
plt.xlim(0, 1000)
plt.show()