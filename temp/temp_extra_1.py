import numpy as np
from matplotlib import pyplot as plt
from utils.utils_image_funcs import shrink, load_image, calc_img_grad, calc_img_divergence, add_white_noise, \
    add_salt_and_pepper_noise

EPSILON = 1e-12


def solve_u_using_gs_combine(u, f, d_x, d_y, b_x, b_y, mu1, mu2, lamda, gs_num_iters, e=0, b_e=0):
    # Hybrid Fidelity Update:
    # mu1: weight for L1 (Salt & Pepper / Outliers)
    # mu2: weight for L2 (Gaussian / Background smoothness)

    # Denominator includes both fidelity terms and the Laplacian relaxation
    const = 1 / (mu1 + mu2 + 4 * lamda)
    diver = calc_img_divergence(d_x - b_x, d_y - b_y)

    # RHS now has two 'pull' forces:
    # 1. mu1*(f + e - b_e) -> The robust median-like estimate
    # 2. mu2*f -> The classic mean-like estimate
    rhs = mu1 * (f + e - b_e) + mu2 * f + lamda * diver

    for _ in range(gs_num_iters):
        # u_sum = np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + \
        #         np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1)
        u_down = np.roll(u, -1, axis=0)
        u_up = np.roll(u, 1, axis=0)
        u_right = np.roll(u, -1, axis=1)
        u_left = np.roll(u, 1, axis=1)
        u_sum = u_down + u_up + u_right + u_left
        # u = const * (lamda * u_sum + rhs)

        G = const * (lamda * u_sum + rhs)
        u = G
    return u


def apply_split_bregman_combine_denoising(f, mu1, mu2, lamda, tolerance, max_iters, is_isotropic, solver_type, gs_num_iters,
                                  show_flag=True):
    # Initialization
    u = np.copy(f)
    d_x, d_y, b_x, b_y = [np.zeros_like(f) for _ in range(4)]

    # Variables for the L1-part of the hybrid fidelity
    e = np.zeros_like(f)
    b_e = np.zeros_like(f)

    normalized_error = np.inf
    normalized_error_vec = []

    for k in range(max_iters):
        u_old = np.copy(u)

        # 1. Update u using the hybrid LHS/RHS
        u = solve_u_using_gs_combine(u, f, d_x, d_y, b_x, b_y, mu1, mu2, lamda, gs_num_iters, e, b_e)

        u_x, u_y = calc_img_grad(u)

        # 2. Update d (Total Variation)
        if is_isotropic:
            s = np.sqrt((u_x + b_x) ** 2 + (u_y + b_y) ** 2 + EPSILON)
            d_x = np.maximum(s - 1 / lamda, 0) * (u_x + b_x) / s
            d_y = np.maximum(s - 1 / lamda, 0) * (u_y + b_y) / s
        else:
            d_x = shrink(u_x + b_x, 1 / lamda)
            d_y = shrink(u_y + b_y, 1 / lamda)

        # 3. Update e (The L1 fidelity 'Median' shrinkage)
        e = shrink(u - f + b_e, 1 / mu1)

        # 4. Update Bregman variables (Adding the noise back)
        b_x += (u_x - d_x)
        b_y += (u_y - d_y)
        b_e += (u - f - e)

        normalized_error = np.linalg.norm(u - u_old) / (np.linalg.norm(u) + EPSILON)
        normalized_error_vec.append(normalized_error)

        if normalized_error < tolerance:
            break

    if show_flag:
        plt.imshow(u, cmap='gray')
        plt.title(f"Hybrid TV-L1-L2 Denoise")
        plt.axis('off')
        plt.show()

    return u, normalized_error_vec


# def add_salt_and_pepper_noise(image, p):
#     noisy_img = np.copy(image)
#     random_matrix = np.random.random(image.shape)
#     noisy_img[random_matrix < (p / 2)] = 0
#     noisy_img[(random_matrix >= (p / 2)) & (random_matrix < p)] = 255
#     return noisy_img


def img_denoise_main():
    image_name = 'Shapes'
    # Recommended hybrid parameters:
    # mu1 high to fight SAP, mu2 low to smooth Gaussian background
    mu1 = 0.01  # L1 weight
    mu2 = 0.001  # L2 weight
    lamda = 0.02  # TV weight
    sigma = 0

    tolerance = 1e-3
    max_iters = 50
    gs_num_iters = 10

    img = load_image(image_name=image_name, show_flag=False)
    # Testing with 20% SAP noise
    noisy_img = add_salt_and_pepper_noise(img, 0.3)
    noisy_img = add_white_noise(image=noisy_img, sigma=sigma, show_flag=False)

    # Note: Passed both mu1 and mu2 to the function
    denoise_img_anisotropic, normalized_error_anisotropic = apply_split_bregman_combine_denoising(
        noisy_img, mu1, mu2, lamda, tolerance, max_iters, False, 'gauss-seidel', gs_num_iters, False)

    denoise_img_isotropic, normalized_error_isotropic = apply_split_bregman_combine_denoising(
        noisy_img, mu1, mu2, lamda, tolerance, max_iters, True, 'gauss-seidel', gs_num_iters, False)

    # # ... Your existing plotting code here (identical to original script) ...
    # # (The axes_cs and axes plots will now show the superior hybrid results)
    #
    # # For demonstration, let's plot the final comparison
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(noisy_img, cmap='gray');
    # axes[0].set_title("Noisy Input (SAP)")
    # axes[1].imshow(denoise_img_anisotropic, cmap='gray');
    # axes[1].set_title("Hybrid Anisotropic")
    # axes[2].imshow(denoise_img_isotropic, cmap='gray');
    # axes[2].set_title("Hybrid Isotropic")
    # plt.show()

    # plot results - denoise
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

    # plt.show()

    # plot results - anisotropic vs. isotropic
    row_idx = 20  # img.shape[0] // 4

    fig_cs, axes_cs = plt.subplots(3, 2, figsize=(14, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plot_data = [
        (noisy_img, "Noisy Image (sigma=15)", "red"),
        (denoise_img_anisotropic, "Anisotropic Denoising", "blue"),
        (denoise_img_isotropic, "Isotropic Denoising", "green")
    ]

    for i, (data_img, title, color) in enumerate(plot_data):
        axes_cs[i, 0].imshow(data_img, cmap='gray')
        axes_cs[i, 0].axhline(y=row_idx, color='yellow', linestyle='--', alpha=0.6)
        axes_cs[i, 0].set_title(title)
        axes_cs[i, 0].axis('off')

        axes_cs[i, 1].plot(data_img[row_idx, :], color=color, linewidth=1.5)
        axes_cs[i, 1].set_title(f"Cross Section (Row {row_idx})")
        axes_cs[i, 1].set_xlabel("Pixel Index")
        axes_cs[i, 1].set_ylabel("Intensity")
        axes_cs[i, 1].set_ylim([-50, 300])
        axes_cs[i, 1].grid(True, alpha=0.3)

    plt.suptitle("Geometric Test Image: Image vs. Intensity Profile", fontsize=16)
    plt.show()


if __name__ == "__main__":
    img_denoise_main()