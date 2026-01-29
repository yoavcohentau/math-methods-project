import cv2
import numpy as np
from matplotlib import pyplot as plt


def calc_img_grad(img):
    img_x = np.roll(img, -1, axis=1) - img
    img_y = np.roll(img, -1, axis=0) - img

    return img_x, img_y


def calc_img_divergence(img_x, img_y):
    div_x = np.roll(img_x, 1, axis=1) - img_x
    div_y = np.roll(img_y, 1, axis=0) - img_y

    return div_x + div_y


def shrink(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def load_image(image_name, show_flag=True):
    if image_name == 'Lena' or image_name == 'Shapes':
        img = cv2.imread(fr'.\images\{image_name}.png', 0)
        img = img.astype(np.float64)
        if show_flag:
            plt.imshow(img, cmap='gray')
            plt.title(f'{image_name} Image')
            plt.axis('off')
            plt.show()
        return img
    else:
        print('No image exists!')


def add_noise(image, sigma, show_flag=True):
    noise = sigma * np.random.randn(*image.shape)
    noisy_img = image + noise
    if show_flag:
        plt.imshow(noisy_img, cmap='gray')
        plt.title(f"Noisy Image")
        plt.axis('off')
        plt.show()
    return noisy_img
