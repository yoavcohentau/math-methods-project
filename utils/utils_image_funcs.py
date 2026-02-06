import cv2
import numpy as np
from matplotlib import pyplot as plt

LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])


def calc_img_grad(img):
    img_x = np.roll(img, -1, axis=1) - img
    img_y = np.roll(img, -1, axis=0) - img

    return img_x, img_y


def calc_img_divergence(img_x, img_y):
    div_x = np.roll(img_x, 1, axis=1) - img_x
    div_y = np.roll(img_y, 1, axis=0) - img_y

    return div_x + div_y


def get_laplacian_kernel_freq_domain(h, w):
    laplacian_kernel_space_domain = np.zeros((h, w))
    laplacian_kernel_space_domain[:3, :3] = LAPLACIAN_KERNEL
    laplacian_kernel_space_domain = np.roll(laplacian_kernel_space_domain, (-1, -1), axis=(0, 1))
    laplacian_kernel_freq_domain = -np.fft.fft2(laplacian_kernel_space_domain)
    return laplacian_kernel_freq_domain


def shrink(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def load_image(image_name, show_flag=True):
    # if image_name == 'Lena' or image_name == 'Shapes':
    try:
        img = cv2.imread(fr'..\images\{image_name}.png', 0)
        img = img.astype(np.float64)
        if show_flag:
            plt.imshow(img, cmap='gray')
            plt.title(f'{image_name} Image')
            plt.axis('off')
            plt.show()
        return img
    except FileNotFoundError as e:
        print(e.errno)
    # else:
    #     print('No image exists!')


def add_white_noise(image, sigma, show_flag=True):
    noise = sigma * np.random.randn(*image.shape)
    noisy_img = image + noise
    if show_flag:
        plt.imshow(noisy_img, cmap='gray')
        plt.title(f"Noisy Image")
        plt.axis('off')
        plt.show()
    return noisy_img


def add_salt_and_pepper_noise(image, p, show_flag=True):
    noisy_img = np.copy(image)
    random_matrix = np.random.random(image.shape)
    noisy_img[random_matrix < (p / 2)] = 0
    noisy_img[(random_matrix >= (p / 2)) & (random_matrix < p)] = 255
    if show_flag:
        plt.imshow(noisy_img, cmap='gray')
        plt.title(f"Noisy Image")
        plt.axis('off')
        plt.show()
    return noisy_img


def create_cs_image(img, mask=None, compress_rate=None, show_flag=True):
    f_full = np.fft.fftshift(np.fft.fft2(img, norm='ortho'))

    if mask is None:
        h, w = img.shape
        mask = np.zeros((h, w))
        num_samples = int(h * w * compress_rate)
        idx = np.random.choice(h * w, num_samples, replace=False)
        mask.flat[idx] = 1

    f_compress = f_full * mask
    u_0 = np.abs(np.fft.ifft2(np.fft.ifftshift(f_compress), norm='ortho'))

    if show_flag:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Original Image ($u$)")

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"Sampling Mask ($R$)\n({int(compress_rate * 100)}% sampled)")

        axes[2].imshow(np.log(np.abs(f_compress) + 1), cmap='magma')
        axes[2].set_title("Sampled K-Space ($f$)\n(Frequency Domain)")

        axes[3].imshow(u_0, cmap='gray')
        axes[3].set_title("Zero-Filled Image ($u_0$)\n(Spatial Domain)")

        for ax in axes:
            ax.axis('off')
        plt.show()

    return f_compress, u_0


def apply_forward_haar_transform(img):
    h, w = img.shape
    h_half = h // 2
    w_half = w // 2
    harr_coeffs = np.zeros_like(img)

    a = img[0:2 * h_half:2, 0:2 * w_half:2]
    b = img[1:2 * h_half:2, 0:2 * w_half:2]
    c = img[0:2 * h_half:2, 1:2 * w_half:2]
    d = img[1:2 * h_half:2, 1:2 * w_half:2]
    harr_coeffs[:h_half, :w_half] = (a + b + c + d) / 2
    harr_coeffs[h_half:2 * h_half, :w_half] = (a - b + c - d) / 2
    harr_coeffs[:h_half, w_half:2 * w_half] = (a + b - c - d) / 2
    harr_coeffs[h_half:2 * h_half, w_half:2 * w_half] = (a - b - c + d) / 2

    return harr_coeffs


def apply_inverse_haar_transform(harr_coeffs):
    h, w = harr_coeffs.shape
    h_half = h // 2
    w_half = w // 2
    img = np.zeros_like(harr_coeffs)

    LL = harr_coeffs[:h_half, :w_half]
    LH = harr_coeffs[h_half:2 * h_half, :w_half]
    HL= harr_coeffs[:h_half, w_half:2 * w_half]
    HH = harr_coeffs[h_half:2 * h_half, w_half:2 * w_half]
    img[0:2 * h_half:2, 0:2 * w_half:2] = (LL + LH + HL + HH) / 2
    img[1:2 * h_half:2, 0:2 * w_half:2] = (LL - LH + HL - HH) / 2
    img[0:2 * h_half:2, 1:2 * w_half:2] = (LL + LH - HL - HH) / 2
    img[1:2 * h_half:2, 1:2 * w_half:2] = (LL - LH - HL + HH) / 2

    return img
