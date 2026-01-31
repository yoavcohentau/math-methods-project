import numpy as np
from matplotlib import pyplot as plt
from utils.utils_image_funcs import shrink, load_image, calc_img_grad, LAPLACIAN_KERNEL

EPSILON = 1e-12


def forward_haar(img):
    """Orthogonal Haar Wavelet Transform W. Handles odd dimensions."""
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    res = np.zeros_like(img)
    # Ensuring even block sizes for slicing
    a = img[0:2 * h2:2, 0:2 * w2:2]
    b = img[1:2 * h2:2, 0:2 * w2:2]
    c = img[0:2 * h2:2, 1:2 * w2:2]
    d = img[1:2 * h2:2, 1:2 * w2:2]
    res[:h2, :w2] = (a + b + c + d) / 2
    res[h2:2 * h2, :w2] = (a - b + c - d) / 2
    res[:h2, w2:2 * w2] = (a + b - c - d) / 2
    res[h2:2 * h2, w2:2 * w2] = (a - b - c + d) / 2
    return res


def inverse_haar(coeffs):
    """Inverse Orthogonal Haar Wavelet Transform W^T."""
    h, w = coeffs.shape
    h2, w2 = h // 2, w // 2
    img = np.zeros_like(coeffs)
    LL, LH = coeffs[:h2, :w2], coeffs[h2:2 * h2, :w2]
    HL, HH = coeffs[:h2, w2:2 * w2], coeffs[h2:2 * h2, w2:2 * w2]
    img[0:2 * h2:2, 0:2 * w2:2] = (LL + LH + HL + HH) / 2
    img[1:2 * h2:2, 0:2 * w2:2] = (LL - LH + HL - HH) / 2
    img[0:2 * h2:2, 1:2 * w2:2] = (LL + LH - HL - HH) / 2
    img[1:2 * h2:2, 1:2 * w2:2] = (LL - LH - HL + HH) / 2
    return img


def solve_cs_split_bregman(f_measured, mask, mu, lamda, gamma, inner_iters, tolerance, max_outer_iters):
    """
    Constrained CS Optimization Algorithm - Strictly following Pages 12-13.
    """
    h, w = f_measured.shape
    num_pixels = h * w

    # --- 1. INITIALIZATION (Page 13, Line 381) ---
    u = np.real(np.fft.ifft2(np.fft.ifftshift(f_measured), norm='ortho'))
    dx = np.zeros_like(u)
    dy = np.zeros_like(u)
    w_coeffs = np.zeros_like(u)
    bx = np.zeros_like(u)
    by = np.zeros_like(u)
    bw = np.zeros_like(u)
    fk = np.copy(f_measured)

    # --- 2. PRE-CALCULATE K (Page 12, Eq 364) ---
    # Constructing the Laplacian kernel precisely
    lap_full = np.zeros((h, w))
    lap_full[:3, :3] = LAPLACIAN_KERNEL
    # Center the kernel to (0,0) to avoid phase artifacts in FFT
    lap_full = np.roll(lap_full, (-1, -1), axis=(0, 1))

    # Frequency response of -Delta (eigenvalues)
    # We take the negative because LAPLACIAN_KERNEL represents Delta, and we need -Delta
    laplace_f = -np.fft.fft2(lap_full)

    # K = mu*R^T*R + lambda*(-Delta_freq) + gamma*I
    # Using fftshift to align Laplacian with measurement mask (k-space center)
    K = mu * mask + lamda * np.fft.fftshift(np.abs(laplace_f)) + gamma
    K[K < 1e-8] = 1

    history = []

    for k in range(max_outer_iters):
        for _ in range(inner_iters):
            # A. Update u (Page 13, Eq 387)
            # Divergence in spatial domain
            div_db = (np.roll(dx - bx, 1, axis=1) - (dx - bx)) + \
                     (np.roll(dy - by, 1, axis=0) - (dy - by))
            w_part = inverse_haar(w_coeffs - bw)

            # Build RHS and move to frequency domain
            rhs = mu * fk + \
                  lamda * np.fft.fftshift(np.fft.fft2(div_db, norm='ortho')) + \
                  gamma * np.fft.fftshift(np.fft.fft2(w_part, norm='ortho'))

            # Direct solve in Fourier domain
            u = np.real(np.fft.ifft2(np.fft.ifftshift(rhs / K), norm='ortho'))

            # B. Update d: Isotropic TV Shrinkage (Eq 388)
            ux, uy = calc_img_grad(u)
            s = np.sqrt(np.abs(ux + bx) ** 2 + np.abs(uy + by) ** 2)
            dx = np.maximum(s - 1 / lamda, 0) * (ux + bx) / (s + EPSILON)
            dy = np.maximum(s - 1 / lamda, 0) * (uy + by) / (s + EPSILON)

            # C. Update w: Wavelet Shrinkage (Eq 390)
            wu = forward_haar(u)
            w_coeffs = shrink(wu + bw, 1 / gamma)

            # D. Update Bregman variables b (Eq 391-393)
            bx += (ux - dx);
            by += (uy - dy);
            bw += (wu - w_coeffs)

        # --- 3. UPDATE fk (Outer Update - Page 13, eq 394) ---
        u_f = np.fft.fftshift(np.fft.fft2(u, norm='ortho'))
        fk += (f_measured - mask * u_f)

        # Monitor convergence
        error = np.linalg.norm(mask * u_f - f_measured) / num_pixels
        history.append(error)
        if error < tolerance: break

    return u, history


# --- Main script ---
img = load_image('MRI', show_flag=False)
img = img / (np.max(img) + EPSILON)

compress_rate = 0.5
h, w = img.shape
mask = np.zeros((h, w))
idx = np.random.choice(h * w, int(h * w * compress_rate), replace=False)
mask.flat[idx] = 1

f_full = np.fft.fftshift(np.fft.fft2(img, norm='ortho'))
f_compress = f_full * mask
u0 = np.abs(np.fft.ifft2(np.fft.ifftshift(f_compress), norm='ortho'))

u_recovered, errors = solve_cs_split_bregman(
    f_measured=f_compress,
    mask=mask,
    mu=1.0,
    lamda=2.0,
    gamma=2.0,
    inner_iters=10,  # Increased for better convergence with precise Laplace
    tolerance=1e-5,
    max_outer_iters=50
)

# Plotting Results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0, 0].imshow(img, cmap='gray');
axes[0, 0].set_title("Original Ground Truth")
axes[0, 1].imshow(u0, cmap='gray');
axes[0, 1].set_title("Zero-Filled Reconstruction")
axes[1, 0].imshow(u_recovered, cmap='gray');
axes[1, 0].set_title("Split Bregman CS (Precise Laplace)")
axes[1, 1].semilogy(errors, color='green', marker='o');
axes[1, 1].set_title("Convergence (Fidelity Residual)")
axes[1, 1].set_xlabel("Outer Iteration (k)");
axes[1, 1].set_ylabel("Error (log scale)")
plt.show()