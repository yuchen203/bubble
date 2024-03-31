import taichi as ti
import numpy as np
import os
import glob
import fm

from utils import *
from project import *

# init taichi
ti.init(arch=ti.cuda, device_memory_GB=4.0, debug=False, default_fp=ti.f32)
# output dir
output_dir = os.path.join("output", "bubble_3d")
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
res_z = 128
L = 2.0 * 1e-3
dx = L / res_y
total_frames = 300
frame_dt = 10.0 * 1e-6
CFL = 0.1
sigma = 58.7 * 1e-3
g = 0.0
narrowband_width = 5.0 * dx
rho_L = 1012.6
rho_G = 12.8

# field
phi = ti.field(dtype=float, shape=(res_x, res_y, res_z))
error = ti.field(dtype=float, shape=(res_x, res_y, res_z))

vel_x = ti.field(dtype=float, shape=(res_x + 1, res_y, res_z))
vel_y = ti.field(dtype=float, shape=(res_x, res_y + 1, res_z))
vel_z = ti.field(dtype=float, shape=(res_x, res_y, res_z + 1))

tmp_field = ti.field(dtype=float, shape=(res_x, res_y, res_z))
tmp_x = ti.field(dtype=float, shape=(res_x + 1, res_y, res_z))
tmp_y = ti.field(dtype=float, shape=(res_x, res_y + 1, res_z))
tmp_z = ti.field(dtype=float, shape=(res_x, res_y, res_z + 1))

max_speed = ti.field(dtype=float, shape=())

# solver
vel_proj = TwoPhaseVelProjJump3d([res_x, res_y, res_z], dx, phi, [vel_x, vel_y, vel_z], rho_L, rho_G)

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
    center1 = ti.Vector([700 * 1e-6, 1000 * 1e-6, 1000 * 1e-6])
    center2 = ti.Vector([1300 * 1e-6, 1000 * 1e-6, 1000 * 1e-6])
    radius = 300 * 1e-6
    for i, j, k in phi:
        pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
        phi[i, j, k] = -ti.min((pos - center1).norm() - radius, (pos - center2).norm() - radius)

def init():
    init_two_spheres_phi_kernel(phi, dx)

def output(frame):
    np.save(os.path.join(phi_dir, f'{frame}.npy'), phi.to_numpy())
    np.save(os.path.join(vel_dir, f'vel_x{frame}.npy'), vel_x.to_numpy())
    np.save(os.path.join(vel_dir, f'vel_y{frame}.npy'), vel_y.to_numpy())
    np.save(os.path.join(vel_dir, f'vel_z{frame}.npy'), vel_z.to_numpy())

@ti.func
def interp_u_MAC(u_x, u_y, u_z, p, dx):
    u_x_p = interp_3d(u_x, p, dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
    u_y_p = interp_3d(u_y, p, dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
    u_z_p = interp_3d(u_z, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)
    return ti.Vector([u_x_p, u_y_p, u_z_p])

@ti.kernel
def advect_field_kernel(
    field_new: ti.template(),
    field_old: ti.template(),
    vel_x: ti.template(),
    vel_y: ti.template(),
    vel_z: ti.template(),
    dx: float,
    dt: float,
    BL_x: float,
    BL_y: float,
    BL_z: float,
):
    neg_dt = -1.0 * dt
    for i, j, k in field_new:
        pos1 = ti.Vector([(i + BL_x) * dx, (j + BL_y) * dx, (k + BL_z) * dx])
        u1 = interp_u_MAC(vel_x, vel_y, vel_z, pos1, dx)
        # first
        pos2 = pos1 + 0.5 * neg_dt * u1
        u2 = interp_u_MAC(vel_x, vel_y, vel_z, pos2, dx)
        # second
        pos3 = pos1 + 0.5 * neg_dt * u2
        u3 = interp_u_MAC(vel_x, vel_y, vel_z, pos3, dx)
        # third
        pos4 = pos1 + neg_dt * u3
        u4 = interp_u_MAC(vel_x, vel_y, vel_z, pos4, dx)

        final_pos = pos1 + neg_dt * 1.0 / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        field_new[i, j, k] = interp_3d(field_old, final_pos, dx, BL_x, BL_y, BL_z)

def fast_marching(phi):
    phi_np = phi.to_numpy().astype(float)
    fm.fm_3d(phi_np, float(narrowband_width), float(dx))
    phi.from_numpy(phi_np)

@ti.kernel
def set_jump_kernel(phi: ti.template(), jump_x: ti.template(), jump_y: ti.template(), jump_z: ti.template(), dt: float):
    for i, j, k in jump_x:
        if i > 0 and i < res_x:
            if phi[i - 1, j, k] < 0.0 and phi[i, j, k] >= 0.0:
                theta = get_theta(phi[i - 1, j, k], phi[i, j, k])
                pos_left = ti.Vector([(i - 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_left + theta * pos_right
                st_jump = sigma * curvature_3d(phi, intf_pos, dx)
                jump_x[i, j, k] = st_jump
            elif phi[i - 1, j, k] >= 0.0 and phi[i, j, k] < 0.0:
                theta = get_theta(phi[i - 1, j, k], phi[i, j, k])
                pos_left = ti.Vector([(i - 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_right + theta * pos_left
                st_jump = sigma * curvature_3d(phi, intf_pos, dx)
                jump_x[i, j, k] = st_jump
            else:
                jump_x[i, j, k] = 0.0
        else:
            jump_x[i, j, k] = 0.0
    
    for i, j, k in jump_y:
        if j > 0 and j < res_y:
            if phi[i, j - 1, k] < 0.0 and phi[i, j, k] >= 0.0:
                theta = get_theta(phi[i, j - 1, k], phi[i, j, k])
                pos_left = ti.Vector([(i + 0.5) * dx, (j - 0.5) * dx, (k + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_left + theta * pos_right
                st_jump = sigma * curvature_3d(phi, intf_pos, dx)
                jump_y[i, j, k] = st_jump
            elif phi[i, j - 1, k] >= 0.0 and phi[i, j, k] < 0.0:
                theta = get_theta(phi[i, j - 1, k], phi[i, j, k])
                pos_left = ti.Vector([(i + 0.5) * dx, (j - 0.5) * dx, (k + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_right + theta * pos_left
                st_jump = sigma * curvature_3d(phi, intf_pos, dx)
                jump_y[i, j, k] = st_jump
            else:
                jump_y[i, j, k] = 0.0
        else:
            jump_y[i, j, k] = 0.0

    for i, j, k in jump_z:
        if k > 0 and k < res_z:
            if phi[i, j, k - 1] < 0.0 and phi[i, j, k] >= 0.0:
                theta = get_theta(phi[i, j, k - 1], phi[i, j, k])
                pos_left = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k - 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_left + theta * pos_right
                st_jump = sigma * curvature_3d(phi, intf_pos, dx)
                jump_z[i, j, k] = st_jump
            elif phi[i, j, k - 1] >= 0.0 and phi[i, j, k] < 0.0:
                theta = get_theta(phi[i, j, k - 1], phi[i, j, k])
                pos_left = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k - 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_right + theta * pos_left
                st_jump = sigma * curvature_3d(phi, intf_pos, dx)
                jump_z[i, j, k] = st_jump
            else:
                jump_z[i, j, k] = 0.0
        else:
            jump_z[i, j, k] = 0.0

def advance(dt):
    advect_field_kernel(tmp_field, phi, vel_x, vel_y, vel_z, dx, dt, 0.5, 0.5, 0.5)
    phi.copy_from(tmp_field)
    fast_marching(phi)

    advect_field_kernel(tmp_x, vel_x, vel_x, vel_y, vel_z, dx, dt, 0.0, 0.5, 0.5)
    advect_field_kernel(tmp_y, vel_y, vel_x, vel_y, vel_z, dx, dt, 0.5, 0.0, 0.5)
    advect_field_kernel(tmp_z, vel_z, vel_x, vel_y, vel_z, dx, dt, 0.5, 0.5, 0.0)
    vel_x.copy_from(tmp_x)
    vel_y.copy_from(tmp_y)
    vel_z.copy_from(tmp_z)

    set_jump_kernel(phi, vel_proj.jump[0], vel_proj.jump[1], vel_proj.jump[2], dt)
    vel_proj.calc_div_kernel(vel_proj.solver.b)

    print("div before proj", np.max(abs(vel_proj.solver.b.to_numpy())))

    vel_proj.project(dt, verbose=True)
    vel_proj.calc_div_kernel(vel_proj.solver.b)
    print("div after proj", np.max(abs(vel_proj.solver.b.to_numpy())))

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
    max_speed[None] = 1.0e-3  # avoid dividing by zero
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        u = 0.5 * (u_x[i, j, k] + u_x[i + 1, j, k])
        v = 0.5 * (u_y[i, j, k] + u_y[i, j + 1, k])
        w = 0.5 * (u_z[i, j, k] + u_z[i, j, k + 1])
        speed = ti.sqrt(u**2 + v**2 + w**2)
        ti.atomic_max(max_speed[None], speed)

def main():
    init()
    output(0)
    for frame in range(1, total_frames):
        print("frame ", frame)
        cur_t = 0.0
        while True:
            calc_max_speed(vel_x, vel_y, vel_z)
            dt = CFL * dx / max_speed[None]
            last_step_in_frame = False
            if cur_t + dt >= frame_dt:
                last_step_in_frame = True
                dt = frame_dt - cur_t
            cur_t += dt

            advance(dt)
            if last_step_in_frame:
                output(frame)
                break


main()