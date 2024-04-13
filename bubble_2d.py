import taichi as ti
import numpy as np
import os
import glob
import fm

import matplotlib.pyplot as plt
from skimage import measure

from utils import *
from project import *


# init taichi
ti.init(arch=ti.cuda, device_memory_GB=4.0, debug=False, default_fp=ti.f32)

# output dir
output_dir = os.path.join("output", "bubble_2d")
phi_dir = os.path.join(output_dir, "phi")
vel_dir = os.path.join(output_dir, "vel")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(phi_dir, exist_ok=True)
os.makedirs(vel_dir, exist_ok=True)


def clean_dir(dir):
    files = glob.glob(os.path.join(dir, "*"))
    for f in files:
        os.remove(f)


clean_dir(phi_dir)
clean_dir(vel_dir)
# hyper parameters
res_x = 128
res_y = 128
L = 2.0 * 1e-3
dx = L / res_y
total_frames = 30
frame_dt = 10.0 * 1e-6
CFL = 0.1
sigma = 58.7 * 1e-3
g = 0.0
narrowband_width = 5.0 * dx
rho_L = 1012.6
rho_G = 12.8
d = 600 * 1e-6
# field
phi = ti.field(dtype=float, shape=(res_x, res_y))
error = ti.field(dtype=float, shape=(res_x, res_y))

vel_x = ti.field(dtype=float, shape=(res_x + 1, res_y))
vel_y = ti.field(dtype=float, shape=(res_x, res_y + 1))

tmp_field = ti.field(dtype=float, shape=(res_x, res_y))
tmp_x = ti.field(dtype=float, shape=(res_x + 1, res_y))
tmp_y = ti.field(dtype=float, shape=(res_x, res_y + 1))

max_speed = ti.field(dtype=float, shape=())

# solver
vel_proj = TwoPhaseVelProjJump2d([res_x, res_y], dx, phi, [vel_x, vel_y], rho_L, rho_G)


@ti.func
def sign(x):
    ret = 0.0
    if x < 0.0:
        ret = -1.0
    elif x == 0.0:
        ret = 0.0
    else:
        ret = 1.0
    return ret


@ti.func
def msign(x):
    ret = 0.0
    if x < 0.0:
        ret = -1.0
    else:
        ret = 1.0
    return ret

@ti.kernel
def init_two_spheres_phi_kernel(phi: ti.template(), dx: float):
    center1 = ti.Vector([700 * 1e-6, 1000 * 1e-6])
    center2 = ti.Vector([1300 * 1e-6, 1000 * 1e-6])
    radius = 300 * 1e-6
    for i, j in phi:
        pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
        phi[i, j] = -ti.min((pos - center1).norm() - radius, (pos - center2).norm() - radius)


def init():
    init_two_spheres_phi_kernel(phi, dx)

def output_phi(phi, dx, output_dir, file_prefix):
    np_phi = phi.to_numpy()
    contours = measure.find_contours(np_phi, level=0)
    # Display the image and plot all contours found
    fig = plt.figure(num=1, figsize=(8, 8), clear=True)
    ax = fig.add_subplot()
    for contour in contours:
        ax.plot(contour[:, 0] * dx, contour[:, 1] * dx, linewidth=2)
    ax.axis("image")
    ax.set_xlim([0.0, L])
    ax.set_ylim([0.0, L])
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, file_prefix + ".jpg"), dpi=512 // 4)


def output_matplot(frame):
    # phi
    output_phi(phi, dx, phi_dir, "{:03d}".format(frame))

    # vel
    write_field(vel_x.to_numpy(), vel_dir, "vel_x{:03d}".format(frame), True)
    write_field(vel_y.to_numpy(), vel_dir, "vel_y{:03d}".format(frame), True)


@ti.func
def interp_u_MAC(u_x, u_y, p, dx):
    u_x_p = interp_2d(u_x, p, dx, BL_x=0.0, BL_y=0.5)
    u_y_p = interp_2d(u_y, p, dx, BL_x=0.5, BL_y=0.0)
    return ti.Vector([u_x_p, u_y_p])


@ti.kernel
def advect_field_kernel(
    field_new: ti.template(),
    field_old: ti.template(),
    vel_x: ti.template(),
    vel_y: ti.template(),
    dx: float,
    dt: float,
    BL_x: float,
    BL_y: float,
):
    for i, j in field_new:
        if i  > 0 and i < res_x - 1 and j > 0 and j < res_y - 1:
            pos = ti.Vector([(i + BL_x) * dx, (j + BL_y) * dx])
            u = interp_u_MAC(vel_x, vel_y, pos, dx)
            partial_x, partial_y = 0.0, 0.0
            # first-order upwind
            if u[0] > 0.0:
                partial_x = (field_old[i, j] - field_old[i - 1, j]) / dx
            else:
                partial_x = (field_old[i + 1, j] - field_old[i, j]) / dx
            if u[1] > 0.0:
                partial_y = (field_old[i, j] - field_old[i, j - 1]) / dx
            else:
                partial_y = (field_old[i, j + 1] - field_old[i, j]) / dx
            field_new[i, j] = field_old[i, j] - dt * (u[0] * partial_x + u[1] * partial_y)


def fast_marching(phi):
    phi_np = phi.to_numpy().astype(float)
    fm.fm_2d(phi_np, float(narrowband_width), float(dx))
    phi.from_numpy(phi_np)


@ti.kernel
def set_jump_kernel(
    phi: ti.template(), jump_x: ti.template(), jump_y: ti.template(), dt: float
):
    for i, j in jump_x:
        if i > 0 and i < res_x:
            if phi[i - 1, j] < 0.0 and phi[i, j] >= 0.0:
                theta = get_theta(phi[i - 1, j], phi[i, j])
                pos_left = ti.Vector([(i - 0.5) * dx, (j + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_left + theta * pos_right
                st_jump = sigma * curvature_2d(phi, intf_pos, dx)
                jump_x[i, j] = st_jump
            elif phi[i - 1, j] >= 0.0 and phi[i, j] < 0.0:
                theta = get_theta(phi[i, j], phi[i - 1, j])
                pos_left = ti.Vector([(i - 0.5) * dx, (j + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_right + theta * pos_left
                st_jump = sigma * curvature_2d(phi, intf_pos, dx)
                jump_x[i, j] = st_jump
            else:
                jump_x[i, j] = 0.0
        else:
            jump_x[i, j] = 0.0

    for i, j in jump_y:
        if j > 0 and j < res_y:
            if phi[i, j - 1] < 0.0 and phi[i, j] >= 0.0:
                theta = get_theta(phi[i, j - 1], phi[i, j])
                pos_bottom = ti.Vector([(i + 0.5) * dx, (j - 0.5) * dx])
                pos_top = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_bottom + theta * pos_top
                st_jump = sigma * curvature_2d(phi, intf_pos, dx)
                jump_y[i, j] = st_jump
            elif phi[i, j - 1] >= 0.0 and phi[i, j] < 0.0:
                theta = get_theta(phi[i, j], phi[i, j - 1])
                pos_bottom = ti.Vector([(i + 0.5) * dx, (j - 0.5) * dx])
                pos_top = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_top + theta * pos_bottom
                st_jump = sigma * curvature_2d(phi, intf_pos, dx)
                jump_y[i, j] = st_jump
            else:
                jump_y[i, j] = 0.0
        else:
            jump_y[i, j] = 0.0


def advance(dt):
    advect_field_kernel(tmp_field, phi, vel_x, vel_y, dx, dt, 0.5, 0.5)
    phi.copy_from(tmp_field)
    fast_marching(phi)

    advect_field_kernel(tmp_x, vel_x, vel_x, vel_y, dx, dt, 0.0, 0.5)
    advect_field_kernel(tmp_y, vel_y, vel_x, vel_y, dx, dt, 0.5, 0.0)
    vel_x.copy_from(tmp_x)
    vel_y.copy_from(tmp_y)

    set_jump_kernel(phi, vel_proj.jump[0], vel_proj.jump[1], dt)
    vel_proj.calc_div_kernel(vel_proj.solver.b)

    print("div before proj", np.max(abs(vel_proj.solver.b.to_numpy())))

    vel_proj.project(dt, verbose=True)
    vel_proj.calc_div_kernel(vel_proj.solver.b)
    print("div after proj", np.max(abs(vel_proj.solver.b.to_numpy())))


@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template()):
    max_speed[None] = 1.0e-3  # avoid dividing by zero
    for i, j in ti.ndrange(res_x, res_y):
        u = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        speed = ti.sqrt(u**2 + v**2)
        ti.atomic_max(max_speed[None], speed)


def main():
    init()
    output_matplot(0)
    for frame in range(1, total_frames):
        print("frame ", frame)
        cur_t = 0.0
        while True:
            calc_max_speed(vel_x, vel_y)
            dt = CFL * dx / max_speed[None]
            last_step_in_frame = False
            if cur_t + dt >= frame_dt:
                last_step_in_frame = True
                dt = frame_dt - cur_t
            cur_t += dt

            advance(dt)
            if last_step_in_frame:
                output_matplot(frame)
                break

def caculate_neck_radius():
    left = res_x // 2 - 1
    right = res_x // 2
    phi_np = phi.to_numpy()
    interface_y = 0.0
    for i in range(res_y // 2 - 1):
        coord_y = res_y // 2 + i
        phi_mid = 0.5 * (phi_np[left, coord_y] + phi_np[right, coord_y])
        phi_next = 0.5 * (phi_np[left, coord_y + 1] + phi_np[right, coord_y + 1])
        if phi_mid * phi_next < 0.0:
            theta = phi_next / (phi_next - phi_mid)
            interface_y = coord_y * theta + (coord_y + 1) * (1.0 - theta)
    pos_y = (interface_y + 0.5) * dx
    neck = pos_y - 0.5 * L
    return neck

def get_simulation_plot():
    init()
    r = caculate_neck_radius()
    r_list = [r / (0.5 * d)]
    t_list = [0.0]
    tau = np.sqrt(rho_L * (0.5 * d) ** 3 / sigma)
    total_time = 0.0
    for frame in range(1, total_frames):
        print("frame ", frame)
        cur_t = 0.0
        while True:
            calc_max_speed(vel_x, vel_y)
            dt = CFL * dx / max_speed[None]
            last_step_in_frame = False
            if cur_t + dt >= frame_dt:
                last_step_in_frame = True
                dt = frame_dt - cur_t
            cur_t += dt
            advance(dt)
            total_time += dt
            if total_time < 0.4 * tau:
                r = caculate_neck_radius()
                r_list.append(r / (0.5 * d))
                t_list.append(total_time / tau)
            if last_step_in_frame:
                output_matplot(frame)
                break
    return t_list, r_list

def f(r):
    global sigma, rho_L, d
    ret = sigma / rho_L
    ret *= 1.0 / (d - np.sqrt(d * d - 4 * r * r)) - 1.0 / (r * 2.0)
    ret = np.sqrt(ret)
    return ret

def solve_thoroddsen():
    global sigma, rho_L, d
    tau = np.sqrt(rho_L * (0.5 * d) ** 3 / sigma)
    num = 100000
    dt = tau / num * 0.4
    r = 6.778e-5
    t_list = [0.0]
    r_list = [r / (0.5 * d)]
    for i in range(num):
        # solve ode with rk4
        r0 = r
        f0 = f(r0)
        r1 = r + f0 * 0.5 * dt
        f1 = f(r1)
        r2 = r + f1 * 0.5 * dt
        f2 = f(r2)
        r3 = r + f2 * dt
        f3 = f(r3)
        r = r + dt / 6 * (f0 + 2 * f1 + 2 * f2 + f3)
        t_list.append((i + 1) * dt / tau)
        r_list.append(r / (0.5 * d))
    return t_list, r_list

def plot():

    # get thoroddsen results
    thoroddsen_t, thoroddsen_r = solve_thoroddsen()
    # get simulation results
    sim_t, sim_r = get_simulation_plot()
    plt.clf()
    plt.plot(thoroddsen_t, thoroddsen_r, label="thoroddsen model")
    plt.plot(sim_t, sim_r, label="simulation")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "neck_radius_evolution.jpg"))

#main()
plot()
