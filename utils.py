import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import numpy as np

eps = 1.0e-6

@ti.func
def sample_2d(qf: ti.template(), u: float, v: float):
    u_dim, v_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim - 1))
    j = ti.max(0, ti.min(int(v), v_dim - 1))
    return qf[i, j]

@ti.func
def N_1(x):
    return 1.0 - ti.abs(x)

@ti.func
def interp_2d(vf, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(0.0, ti.min(u, u_dim - 1 - eps))
    t = ti.max(0.0, ti.min(v, v_dim - 1 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample_2d(vf, iu, iv)
    b = sample_2d(vf, iu + 1, iv)
    c = sample_2d(vf, iu, iv + 1)
    d = sample_2d(vf, iu + 1, iv + 1)

    interped = (
        a * N_1(fu) * N_1(fv)
        + b * N_1(fu - 1) * N_1(fv)
        + c * N_1(fu) * N_1(fv - 1)
        + d * N_1(fu - 1) * N_1(fv - 1)
    )

    return interped

@ti.func
def get_theta(phi0, phi1):
    non_neg_phi = 0.0
    neg_phi = 0.0
    if phi0 < 0.0 and phi1 >= 0.0:
        neg_phi = phi0
        non_neg_phi = phi1
    elif phi0 >= 0.0 and phi1 < 0.0:
        neg_phi = phi1
        non_neg_phi = phi0
    else:
        print("Error in theta: phi0 and phi1 have the same sign")
    return neg_phi / (neg_phi - non_neg_phi)

@ti.func
def normal_2d(phi, pos, dx):
    nx = (interp_2d(phi, pos + ti.Vector([dx, 0.0], dt=float), dx) - interp_2d(phi, pos - ti.Vector([dx, 0.0], dt=float), dx)) / (2 * dx)
    ny = (interp_2d(phi, pos + ti.Vector([0.0, dx], dt=float), dx) - interp_2d(phi, pos - ti.Vector([0.0, dx], dt=float), dx)) / (2 * dx)
    return ti.Vector([nx, ny], dt=float).normalized()

@ti.func
def curvature_2d(phi, pos, dx):
    one_over_dx = 1.0 / dx
    one_over_2dx = 0.5 * one_over_dx
    nx_left = normal_2d(phi, pos - ti.Vector([dx, 0.0], dt=float), dx)
    nx_right = normal_2d(phi, pos + ti.Vector([dx, 0.0], dt=float), dx)
    ny_left = normal_2d(phi, pos - ti.Vector([0.0, dx], dt=float), dx)
    ny_right = normal_2d(phi, pos + ti.Vector([0.0, dx], dt=float), dx)
    ret = (nx_right[0] - nx_left[0] + ny_right[1] - ny_left[1]) * one_over_2dx
    if abs(ret) > one_over_dx:
        if ret < 0.0:
            ret = -one_over_dx
        else:
            ret = one_over_dx
    return ret

def write_field(
    img, outdir, file_prefix, relative=False, vmin=0.0, vmax=1.0, grayscale=False
):
    array = img[:, :, np.newaxis]
    array = np.transpose(array, (1, 0, 2))  # from X,Y to Y, X
    x_to_y = array.shape[1] / array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=(x_to_y * y_size + 1, y_size), clear=True)
    ax = fig.add_subplot()
    # fig.subplots_adjust(0.1, 0.1, 0.9, 0.9)
    # ax.set_axis_off()
    ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([0, array.shape[0]])
    cmap = "jet"
    if grayscale:
        cmap = "Greys"
    if relative:
        p = ax.imshow(array, alpha=0.4, cmap=cmap)
    else:
        p = ax.imshow(array, alpha=0.4, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(outdir, file_prefix + ".jpg"), dpi=512 // 4)
