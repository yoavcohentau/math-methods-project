import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def calc_img_grad(img):
    img_x = np.roll(img, -1, axis=1) - img
    img_y = np.roll(img, -1, axis=0) - img

    return img_x, img_y


def calc_img_divergence(img_x, img_y):
    # div_x = img_x - np.roll(img_x, 1, axis=1)
    # div_y = img_y - np.roll(img_y, 1, axis=0)
    div_x = np.roll(img_x, 1, axis=1) - img_x
    div_y = np.roll(img_y, 1, axis=0) - img_y

    return div_x + div_y


def solve_u_using_gs(u, f, d_x, d_y, b_x, b_y, mu, lamda, gs_num_iters):
    const = 1 / (mu+4*lamda)
    diver = calc_img_divergence(d_x-b_x, d_y-b_y)
    rhs = mu*f + lamda*diver

    for _ in range(gs_num_iters):
        u_down = np.roll(u, -1, axis=0)
        u_up = np.roll(u, 1, axis=0)
        u_right = np.roll(u, -1, axis=1)
        u_left = np.roll(u, 1, axis=1)
        u_sum = u_down + u_up + u_right + u_left

        G = const * (lamda*u_sum + rhs)
        u = G



    # dx_left = np.roll(d_x, 1, axis=0)
    # dy_up = np.roll(d_y, 1, axis=1)
    # d_sum = (dx_left - d_x) + (dy_up - d_y)
    #
    # bx_left = np.roll(b_x, 1, axis=0)
    # by_up = np.roll(b_y, 1, axis=1)
    # b_sum = (bx_left - b_x) - (by_up - b_y)
    #
    # for _ in range(gs_num_iters):
    #     u_down = np.roll(u, -1, axis=0)
    #     u_up = np.roll(u, 1, axis=0)
    #     u_right = np.roll(u, -1, axis=1)
    #     u_left = np.roll(u, 1, axis=1)
    #     u_sum = u_down + u_up + u_right + u_left
    #
    #     G = const * (lamda*(u_sum + d_sum + b_sum) + mu*f)
    #     u = G

    return u


def apply_split_bregman_denoising(f, mu, lamda, tolerance, max_iters, is_isotropic, solver_type, gs_num_iters, show_flag=True):
    # initialization
    u = np.copy(f)
    d_x = np.zeros_like(f)
    d_y = np.zeros_like(f)
    b_x = np.zeros_like(f)
    b_y = np.zeros_like(f)

    normalized_error = np.inf
    history = []

    while normalized_error > tolerance:
        u_new = solve_u_using_gs(u, f, d_x, d_y, b_x, b_y, mu, lamda, gs_num_iters)

        u_x, u_y = calc_img_grad(u_new)

        if is_isotropic:
            s = np.sqrt((u_x + b_x) ** 2 + (u_y + b_y) ** 2)
            d_x = np.maximum(s - 1 / lamda, 0) * (u_x + b_x) / (s + 1e-12)
            d_y = np.maximum(s - 1 / lamda, 0) * (u_y + b_y) / (s + 1e-12)
        else:
            d_x = np.sign(u_x + b_x) * np.maximum(np.abs(u_x + b_x) - 1 / lamda, 0)
            d_y = np.sign(u_y + b_y) * np.maximum(np.abs(u_y + b_y) - 1 / lamda, 0)

        b_x += (u_x - d_x)
        b_y += (u_y - d_y)
        normalized_error = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        history.append(normalized_error)

        u = u_new

    if show_flag:
        plt.imshow(u, cmap='gray')
        plt.title(f"Denoise Image")
        plt.axis('off')
        plt.show()

    return u, history


def img_denoise_main():
    image_name = 'Shapes'  # 'Shapes' or 'Lena'

    sigma = 15

    mu = 0.05
    lamda = 0.1
    tolerance = 5e-3
    max_iters = 50
    is_isotropic = False
    solver_type = 'gauss-seidel'
    gs_num_iters = 10

    img = load_image(image_name=image_name, show_flag=False)
    noisy_img = add_noise(image=img, sigma=sigma, show_flag=False)
    denoise_img = apply_split_bregman_denoising(noisy_img, mu, lamda, tolerance, max_iters, is_isotropic, solver_type, gs_num_iters)
    pass


if __name__ == "__main__":
    img_denoise_main()
