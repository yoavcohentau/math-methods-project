import numpy as np
from matplotlib import pyplot as plt

from utils.utils_image_funcs import shrink, load_image, add_noise, calc_img_grad, calc_img_divergence

EPSILON = 1e-12


def solve_u_using_gs(u, f, d_x, d_y, b_x, b_y, mu, lamda, gs_num_iters):
    const = 1 / (mu + 4 * lamda)
    diver = calc_img_divergence(d_x - b_x, d_y - b_y)
    rhs = mu * f + lamda * diver

    for _ in range(gs_num_iters):
        u_down = np.roll(u, -1, axis=0)
        u_up = np.roll(u, 1, axis=0)
        u_right = np.roll(u, -1, axis=1)
        u_left = np.roll(u, 1, axis=1)
        u_sum = u_down + u_up + u_right + u_left

        G = const * (lamda * u_sum + rhs)
        u = G

    return u


def apply_split_bregman_denoising(f, mu, lamda, tolerance, max_iters, is_isotropic, solver_type, gs_num_iters,
                                  show_flag=True):
    # initialization
    u = np.copy(f)
    d_x = np.zeros_like(f)
    d_y = np.zeros_like(f)
    b_x = np.zeros_like(f)
    b_y = np.zeros_like(f)

    normalized_error = np.inf
    normalized_error_vec = []

    while normalized_error > tolerance:
        u_new = solve_u_using_gs(u, f, d_x, d_y, b_x, b_y, mu, lamda, gs_num_iters)

        u_x, u_y = calc_img_grad(u_new)

        if is_isotropic:
            s = np.sqrt((u_x + b_x) ** 2 + (u_y + b_y) ** 2)
            d_x = np.maximum(s - 1 / lamda, 0) * (u_x + b_x) / (s + EPSILON)
            d_y = np.maximum(s - 1 / lamda, 0) * (u_y + b_y) / (s + EPSILON)
        else:
            d_x = shrink(u_x + b_x, 1 / lamda)
            d_y = shrink(u_y + b_y, 1 / lamda)

        b_x += (u_x - d_x)
        b_y += (u_y - d_y)
        normalized_error = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        normalized_error_vec.append(normalized_error)

        u = u_new

    if show_flag:
        plt.imshow(u, cmap='gray')
        plt.title(f"Denoise Image")
        plt.axis('off')
        plt.show()

    return u, normalized_error_vec


def img_denoise_main():
    image_name = 'Shapes'  # 'Shapes' or 'Lena'

    sigma = 20

    mu = 0.01 #0.05
    lamda = 0.02 #0.1
    tolerance = 1e-3
    max_iters = 50
    is_isotropic = False
    solver_type = 'gauss-seidel'
    gs_num_iters = 10

    img = load_image(image_name=image_name, show_flag=False)
    noisy_img = add_noise(image=img, sigma=sigma, show_flag=False)

    denoise_img_anisotropic, normalized_error_anisotropic = apply_split_bregman_denoising(f=noisy_img,
                                                                                          mu=mu,
                                                                                          lamda=lamda,
                                                                                          tolerance=tolerance,
                                                                                          max_iters=max_iters,
                                                                                          is_isotropic=False,
                                                                                          solver_type=solver_type,
                                                                                          gs_num_iters=gs_num_iters,
                                                                                          show_flag=False)
    denoise_img_isotropic, normalized_error_isotropic = apply_split_bregman_denoising(f=noisy_img,
                                                                                      mu=mu,
                                                                                      lamda=lamda,
                                                                                      tolerance=tolerance,
                                                                                      max_iters=max_iters,
                                                                                      is_isotropic=True,
                                                                                      solver_type=solver_type,
                                                                                      gs_num_iters=gs_num_iters,
                                                                                      show_flag=False)

    # plot results
    fig = plt.figure(figsize=(14, 18))
    plt.subplots_adjust(hspace=0.4)

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original Clean Image")
    ax1.axis('off')

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.imshow(noisy_img, cmap='gray')
    ax2.set_title(f"Noisy Image (sigma={sigma})")
    ax2.axis('off')

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.imshow(denoise_img_anisotropic, cmap='gray')
    ax3.set_title("Anisotropic Denoising")
    ax3.axis('off')

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.imshow(denoise_img_isotropic, cmap='gray')
    ax4.set_title("Isotropic Denoising")
    ax4.axis('off')

    ax5 = fig.add_subplot(3, 1, 3)
    ax5.semilogy(normalized_error_anisotropic, label='Anisotropic', color='red', linewidth=2)
    ax5.semilogy(normalized_error_isotropic, label='Isotropic', color='blue', linewidth=2, linestyle='--')
    ax5.set_title("Convergence Comparison (Normalized Error)")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Error (log scale)")
    ax5.legend()
    ax5.grid(True, which="both", ls="-", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    img_denoise_main()
