import numpy as np
from matplotlib import pyplot as plt
from utils.utils_image_funcs import shrink, load_image, calc_img_grad, get_laplacian_kernel_freq_domain, \
    calc_img_divergence, apply_forward_haar_transform, apply_inverse_haar_transform, create_cs_image

EPSILON = 1e-12


def solve_cs_split_bregman(f_measured, mask, mu, lamda, gamma, inner_iters, tolerance, max_outer_iters):
    h, w = f_measured.shape
    num_pixels = h * w

    # initialization
    u = np.real(np.fft.ifft2(np.fft.ifftshift(f_measured), norm='ortho'))
    d_x = np.zeros_like(u)
    d_y = np.zeros_like(u)
    w_coeffs = np.zeros_like(u)
    b_x = np.zeros_like(u)
    b_y = np.zeros_like(u)
    b_w = np.zeros_like(u)
    f_k = np.copy(f_measured)

    # calc K kernel
    laplacian_kernel_freq = get_laplacian_kernel_freq_domain(h, w)
    K = mu * mask + lamda * np.fft.fftshift(np.abs(laplacian_kernel_freq)) + gamma
    K[K < 1e-8] = 1

    err_vec = []
    for k in range(max_outer_iters):
        for _ in range(inner_iters):
            # rhs
            div_db = calc_img_divergence(d_x - b_x, d_y - b_y)
            w_part = apply_inverse_haar_transform(w_coeffs - b_w)
            rhs = mu * f_k + np.fft.fftshift(np.fft.fft2(lamda*div_db + gamma*w_part, norm='ortho'))

            # solve u
            u = np.real(np.fft.ifft2(np.fft.ifftshift(rhs / K), norm='ortho'))

            # update d
            u_x, u_y = calc_img_grad(u)
            s = np.sqrt(np.abs(u_x + b_x) ** 2 + np.abs(u_y + b_y) ** 2)
            d_x = np.maximum(s - 1 / lamda, 0) * (u_x + b_x) / (s + EPSILON)
            d_y = np.maximum(s - 1 / lamda, 0) * (u_y + b_y) / (s + EPSILON)

            # update w
            wu = apply_forward_haar_transform(u)
            w_coeffs = shrink(wu + b_w, 1 / gamma)

            # update b
            b_x += (u_x - d_x)
            b_y += (u_y - d_y)
            b_w += (wu - w_coeffs)

        # update f_k for constrained algorithm
        u_f = np.fft.fftshift(np.fft.fft2(u, norm='ortho'))
        f_k += (f_measured - mask * u_f)

        # calc error for constrained algorithm
        error = np.linalg.norm(mask * u_f - f_measured) / num_pixels
        err_vec.append(error)
        if error < tolerance:
            break

    return u, err_vec


img = load_image('MRI', show_flag=False)
img = img / (np.max(img) + EPSILON)

compress_rate = 0.3
h, w = img.shape
mask = np.zeros((h, w))
idx = np.random.choice(h * w, int(h * w * compress_rate), replace=False)
mask.flat[idx] = 1

# keep the center of the k-space (low frequencies)
center_size = 1
mask[h//2-center_size:h//2+center_size, w//2-center_size:w//2+center_size] = 1

f_compress, u_0 = create_cs_image(img=img, mask=mask, compress_rate=compress_rate)

u_recovered, errors = solve_cs_split_bregman(
    f_measured=f_compress,
    mask=mask,
    mu=2.0,
    lamda=4.0,
    gamma=4.0,
    inner_iters=10,
    tolerance=1e-5,
    max_outer_iters=50
)

# Plotting Results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original Ground Truth")
axes[0, 1].imshow(u_0, cmap='gray')
axes[0, 1].set_title("Zero-Filled Reconstruction")
axes[1, 0].imshow(u_recovered, cmap='gray')
axes[1, 0].set_title("Split Bregman CS (Precise Laplace)")
axes[1, 1].semilogy(errors, color='green', marker='o')
axes[1, 1].set_title("Convergence (Fidelity Residual)")
axes[1, 1].set_xlabel("Outer Iteration (k)")
axes[1, 1].set_ylabel("Error (log scale)")
plt.show()
