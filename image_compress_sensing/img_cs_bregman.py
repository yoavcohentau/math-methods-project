from utils.utils_image_funcs import load_image, create_cs_image

import numpy as np

import numpy as np
from matplotlib import pyplot as plt
from utils.utils_image_funcs import shrink, load_image, calc_img_grad

EPSILON = 1e-12


def solve_cs_split_bregman(f_measured, mask, mu, lamda, inner_iters, tolerance, max_outer_iters):
    h, w = f_measured.shape

    # Use orthogonal normalization ('ortho') to keep scales consistent between spatial and frequency domains
    u = np.real(np.fft.ifft2(np.fft.ifftshift(f_measured), norm='ortho'))
    dx = np.zeros_like(u);
    dy = np.zeros_like(u)
    bx = np.zeros_like(u);
    by = np.zeros_like(u)

    fk = np.copy(f_measured)

    freq_r = np.fft.fftfreq(h).reshape(-1, 1)
    freq_c = np.fft.fftfreq(w).reshape(1, -1)
    laplace_kernel = 4 - 2 * np.cos(2 * np.pi * freq_r) - 2 * np.cos(2 * np.pi * freq_c)

    # Denominator K matches the circulant structure (Page 12, eq 364)
    K = mu * mask + lamda * laplace_kernel
    K[K == 0] = 1

    normalized_error_vec = []
    outer_k = 0

    while (outer_k < max_outer_iters):
        u_old_outer = np.copy(u)

        for _ in range(inner_iters):
            # Divergence in spatial domain
            div_spatial = (np.roll(dx - bx, 1, axis=1) - (dx - bx)) + \
                          (np.roll(dy - by, 1, axis=0) - (dy - by))

            # Important: Use norm='ortho' here too
            rhs = mu * fk + lamda * np.fft.fftshift(np.fft.fft2(div_spatial, norm='ortho'))
            u_freq = rhs / K

            u = np.real(np.fft.ifft2(np.fft.ifftshift(u_freq), norm='ortho'))

            ux, uy = calc_img_grad(u)

            # Isotropic TV shrinkage (Page 10, eq 4.4)
            s = np.sqrt(np.abs(ux + bx) ** 2 + np.abs(uy + by) ** 2)
            dx = np.maximum(s - 1 / lamda, 0) * (ux + bx) / (s + EPSILON)
            dy = np.maximum(s - 1 / lamda, 0) * (uy + by) / (s + EPSILON)

            bx += (ux - dx)
            by += (uy - dy)

        # Update measurements with 'ortho' consistency
        u_f = np.fft.fftshift(np.fft.fft2(u, norm='ortho'))
        fk += (f_measured - mask * u_f)

        error = np.linalg.norm(u - u_old_outer) / (np.linalg.norm(u) + EPSILON)
        normalized_error_vec.append(error)
        outer_k += 1
        if error < tolerance: break

    return u, normalized_error_vec


img = load_image('MRI', show_flag=False)
compress_rate = 0.3

h, w = img.shape
mask = np.zeros((h, w))
num_samples = int(h * w * compress_rate)
idx = np.random.choice(h * w, num_samples, replace=False)
mask.flat[idx] = 1

f_compress, u_0 = create_cs_image(img=img, mask=mask, compress_rate=compress_rate)

u, normalized_error_vec = solve_cs_split_bregman(f_measured=f_compress,
                                                 mask=mask,
                                                 mu=1.0,
                                                 lamda=2.0,
                                                 inner_iters=5,
                                                 tolerance=1e-4,
                                                 max_outer_iters=30)

# --- Plotting Results ---

# 1. Create a figure for visual comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.3)

# Row 1, Left: Original Ground Truth
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original Image (Ground Truth)")
axes[0, 0].axis('off')

# Row 1, Right: Initial Zero-filled Reconstruction
# This is what the image looks like before Split Bregman processing
axes[0, 1].imshow(u_0, cmap='gray')
axes[0, 1].set_title(f"Zero-Filled Reconstruction\n({int(compress_rate*100)}% Measurements)")
axes[0, 1].axis('off')

# Row 2, Left: Final Recovered Image
# The output of the Split Bregman CS algorithm [cite: 380, 525]
axes[1, 0].imshow(u, cmap='gray')
axes[1, 0].set_title("Split Bregman CS Recovery\n(TV Regularized)")
axes[1, 0].axis('off')

# Row 2, Right: Convergence Graph
# Monitoring the normalized error over outer iterations [cite: 414]
axes[1, 1].semilogy(normalized_error_vec, color='green', marker='o', linewidth=2)
axes[1, 1].set_title("Convergence Comparison (Outer Loop)")
axes[1, 1].set_xlabel("Outer Iteration (k)")
axes[1, 1].set_ylabel("Normalized Error (log scale)")
axes[1, 1].grid(True, which="both", ls="-", alpha=0.5)

plt.show()

# 2. Display the Sampling Mask (R) separately to see the sparsity pattern [cite: 326]
plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap='gray')
plt.title(f"Random Sampling Mask (Rate: {compress_rate})")
plt.axis('off')
plt.show()
