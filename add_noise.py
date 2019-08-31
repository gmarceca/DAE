import numpy as np

def noisy(image):
    print('min {} , max {} and mean {}'.format(np.min(image), np.max(image), np.mean(image)))
    p = 0.5
    noisy = np.random.binomial(1, p,image.shape)
    noisy = image*noisy
    print('min {} , max {} and mean {}'.format(np.min(noisy), np.max(noisy), np.mean(noisy)))
    return noisy
