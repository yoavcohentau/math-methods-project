from matplotlib import pyplot as plt

from image_denoising.split_bregman_denoisers.regular_bregman_denoising import apply_split_bregman_denoising
from image_denoising.split_bregman_denoisers.median_bregman_denoising import apply_split_bregman_median_denoising
from image_denoising.split_bregman_denoisers.combine_bregman_denoising import apply_split_bregman_combine_denoising
from utils.utils_image_funcs import load_image, add_white_noise, add_salt_and_pepper_noise


def img_denoise_main():
    image_name = 'Shapes'  # 'Shapes' or 'Lena'

    # noise params
    sigma_white_noise = 20
    p_salt_n_paper = 0.0

    # algo params
    algorithm_type = "regular"  # "regular" ot "median" or "combine"
    mu = 0.01 #0.05
    lamda = 0.02 #0.1
    tolerance = 1e-3
    max_iters = 50
    solver_type = 'gauss-seidel'
    gs_num_iters = 10

    img = load_image(image_name=image_name, show_flag=False)
    noisy_img = add_white_noise(image=img, sigma=sigma_white_noise, show_flag=False)
    noisy_img = add_salt_and_pepper_noise(image=noisy_img, p=p_salt_n_paper)

    if algorithm_type == "regular":
        denoise_img_anisotropic, normalized_error_anisotropic = apply_split_bregman_denoising(
            f=noisy_img,
            mu=mu,
            lamda=lamda,
            tolerance=tolerance,
            max_iters=max_iters,
            is_isotropic=False,
            solver_type=solver_type,
            gs_num_iters=gs_num_iters,
            show_flag=False)
        denoise_img_isotropic, normalized_error_isotropic = apply_split_bregman_denoising(
            f=noisy_img,
            mu=mu,
            lamda=lamda,
            tolerance=tolerance,
            max_iters=max_iters,
            is_isotropic=True,
            solver_type=solver_type,
            gs_num_iters=gs_num_iters,
            show_flag=False)

    elif algorithm_type == "median":
        denoise_img_anisotropic, normalized_error_anisotropic = apply_split_bregman_median_denoising(
            f=noisy_img,
            mu=mu,
            lamda=lamda,
            tolerance=tolerance,
            max_iters=max_iters,
            is_isotropic=False,
            solver_type=solver_type,
            gs_num_iters=gs_num_iters,
            show_flag=False)
        denoise_img_isotropic, normalized_error_isotropic = apply_split_bregman_median_denoising(
            f=noisy_img,
            mu=mu,
            lamda=lamda,
            tolerance=tolerance,
            max_iters=max_iters,
            is_isotropic=True,
            solver_type=solver_type,
            gs_num_iters=gs_num_iters,
            show_flag=False)

    elif algorithm_type == "combine":
        mu1 = mu
        mu2 = mu / 10
        denoise_img_anisotropic, normalized_error_anisotropic = apply_split_bregman_combine_denoising(
            f=noisy_img,
            mu1=mu1,
            mu2=mu2,
            lamda=lamda,
            tolerance=tolerance,
            max_iters=max_iters,
            is_isotropic=False,
            solver_type=solver_type,
            gs_num_iters=gs_num_iters,
            show_flag=False)
        denoise_img_isotropic, normalized_error_isotropic = apply_split_bregman_combine_denoising(
            f=noisy_img,
            mu1=mu1,
            mu2=mu2,
            lamda=lamda,
            tolerance=tolerance,
            max_iters=max_iters,
            is_isotropic=True,
            solver_type=solver_type,
            gs_num_iters=gs_num_iters,
            show_flag=False)
    else:
        raise NotImplementedError()

    # plot results - denoise
    fig = plt.figure(figsize=(14, 18))
    plt.subplots_adjust(hspace=0.4)

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original Clean Image")
    ax1.axis('off')

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.imshow(noisy_img, cmap='gray')
    ax2.set_title(f"Noisy Image (sigma={sigma_white_noise}, p={p_salt_n_paper})")
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
        (noisy_img, f"Noisy Image (sigma={sigma_white_noise}, p={p_salt_n_paper})", "red"),
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
