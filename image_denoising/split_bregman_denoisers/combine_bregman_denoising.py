import numpy as np
from matplotlib import pyplot as plt

from utils.utils_image_funcs import calc_img_divergence, calc_img_grad, shrink

EPSILON = 1e-12


def solve_u_using_gs_combine(u, f, d_x, d_y, b_x, b_y, mu1, mu2, lamda, gs_num_iters, e=0, b_e=0):
    const = 1 / (mu1 + mu2 + 4 * lamda)
    diver = calc_img_divergence(d_x - b_x, d_y - b_y)
    rhs = mu1 * (f + e - b_e) + mu2 * f + lamda * diver

    for _ in range(gs_num_iters):
        u_down = np.roll(u, -1, axis=0)
        u_up = np.roll(u, 1, axis=0)
        u_right = np.roll(u, -1, axis=1)
        u_left = np.roll(u, 1, axis=1)
        u_sum = u_down + u_up + u_right + u_left

        G = const * (lamda * u_sum + rhs)
        u = G

    return u


def apply_split_bregman_combine_denoising(f, mu1, mu2, lamda, tolerance, max_iters, is_isotropic, solver_type, gs_num_iters,
                                  show_flag=True):
    # initialization
    u = np.copy(f)
    d_x = np.zeros_like(f)
    d_y = np.zeros_like(f)
    b_x = np.zeros_like(f)
    b_y = np.zeros_like(f)

    # The new L1 constrain - the "median"
    e = np.zeros_like(f)
    b_e = np.zeros_like(f)

    normalized_error_vec = []
    for k in range(max_iters):
        u_new = solve_u_using_gs_combine(u, f, d_x, d_y, b_x, b_y, mu1, mu2, lamda, gs_num_iters, e, b_e)

        u_x, u_y = calc_img_grad(u_new)

        # 2. Update d (Total Variation)
        if is_isotropic:
            s = np.sqrt((u_x + b_x) ** 2 + (u_y + b_y) ** 2 + EPSILON)
            d_x = np.maximum(s - 1 / lamda, 0) * (u_x + b_x) / s
            d_y = np.maximum(s - 1 / lamda, 0) * (u_y + b_y) / s
        else:
            d_x = shrink(u_x + b_x, 1 / lamda)
            d_y = shrink(u_y + b_y, 1 / lamda)

        e = shrink(u - f + b_e, 1 / mu1)

        b_x += (u_x - d_x)
        b_y += (u_y - d_y)
        b_e += (u - f - e)

        normalized_error = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + EPSILON)
        normalized_error_vec.append(normalized_error)

        u = u_new

        if normalized_error < tolerance:
            break

    if show_flag:
        plt.imshow(u, cmap='gray')
        plt.title(f"Denoise Image")
        plt.axis('off')
        plt.show()

    return u, normalized_error_vec
