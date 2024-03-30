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
ti.init(arch=ti.cuda, device_memory_GB=4.0, debug=False, default_fp=ti.f64)

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
dx = 1.0 / res_y
total_frames = 100
frame_dt = 0.01
CFL = 0.2
sigma = 1.0
g = 0.0
narrowband_width = 5.0 * dx
rho_ratio = 10.0
# field
phi = ti.field(dtype=float, shape = (res_x, res_y))
error = ti.field(dtype = float, shape = (res_x, res_y))

vel_x = ti.field(dtype =float, shape = (res_x + 1, res_y))
vel_y = ti.field(dtype =float, shape = (res_x, res_y + 1))

tmp_field = ti.field(dtype = float, shape = (res_x, res_y))
tmp_x = ti.field(dtype = float, shape = (res_x + 1, res_y))
tmp_y = ti.field(dtype = float, shape = (res_x, res_y + 1))

max_speed = ti.field(dtype = float, shape = ())

# solver
vel_proj = TwoPhaseVelProjJump2d([res_x, res_y], dx, phi, [vel_x, vel_y], rho_ratio)
@ti.func
def sign(x):
    ret = 0.0
    if x <0.0:
        ret = -1.0
    elif x ==0.0:
        ret = 0.0
    else:
        ret = 1.0
    return ret

@ti.func
def msign(x):
    ret = 0.0
    if x <0.0:
        ret = -1.0
    else:
        ret = 1.0
    return ret

@ti.kernel
def init_ellipse_phi_kernel(phi:ti.template(), dx:float):
    center = ti.Vector([0.5, 0.5])
    a = 0.3
    b = 0.1
    for i, j in phi:
        pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
        dis = ti.abs(pos - center)
        p = dis
        ab = ti.Vector([a, b])
        if p[0] > p[1]:
            p = ti.Vector([p[1], p[0]])
            ab = ti.Vector([b, a])
        l = ab[1] * ab[1] - ab[0] * ab[0]
        m = ab[0] * p[0] / l
        n = ab[1] * p[1] / l
        m2 = m * m
        n2 = n * n
        c = (m2 + n2 - 1.0) / 3.0
        c3 = c * c * c
        d = c3 + m2 *n2
        q = d + m2 *n2
        g = m + m*n2
        co = 0.0
        if d<0.0:
            h = ti.acos(q/c3)/3.0
            s = ti.cos(h) + 2.0
            t = ti.sin(h) * ti.sqrt(3.0)
            rx  = ti.sqrt(m2 - c*(s+t))
            ry= ti.sqrt(m2 - c*(s-t))
            co = ry + sign(l) * rx + ti.abs(g) / (rx*ry)
        else:
            h = 2.0 * m * n * ti.sqrt(d)
            s = msign(q + h) *ti.pow(ti.abs(q + h), 1.0/3.0)
            t = msign(q - h) *ti.pow(ti.abs(q - h), 1.0/3.0)
            rx = -(s + t) - c * 4.0 + 2.0 * m2
            ry = (s - t) * ti.sqrt(3.0)
            rm = ti.sqrt(rx * rx + ry * ry)
            co = ry / ti.sqrt(rm - rx) + 2.0 * g / rm
        co=(co-m)/2.0
        si = ti.sqrt(ti.max(1.0-co*co, 0.0))
        r = ab *ti.Vector([co, si])
        phi[i, j] = (r-p).norm() * msign(p[1] - r[1])
        
@ti.kernel
def init_sphere_phi(phi: ti.template(), dx: float):
    center = ti.Vector([0.5, 0.5])
    radius = 0.3
    for i, j in phi:
        pos = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
        phi[i, j] = (pos - center).norm() - radius

@ti.kernel
def init_drop_tank_phi_kernel(
    phi: ti.template(),
    dx: float,
):
    r = 0.1
    for i, j in phi:
        dis = ti.sqrt(((i + 0.5) * dx - 0.5) ** 2 + ((j + 0.5) * dx - 0.75) ** 2)
        phi[i, j] = dis - r
        phi[i, j] = ti.min((j + 0.5) * dx - 0.35, phi[i, j])

def init():
    init_ellipse_phi_kernel(phi, dx)
    #init_drop_tank_phi_kernel(phi, dx)
    #init_sphere_phi(phi, dx)

def output_phi(phi, dx, output_dir, file_prefix):
    np_phi = phi.to_numpy()
    contours = measure.find_contours(np_phi, level=0)
    # Display the image and plot all contours found
    fig = plt.figure(num=1, figsize=(8, 8), clear=True)
    ax = fig.add_subplot()
    for contour in contours:
        ax.plot(contour[:, 0] * dx, contour[:, 1] * dx, linewidth=2)
    ax.axis("image")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
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
    neg_dt = -1.0 * dt
    for i, j in field_new:
        pos = ti.Vector([(i + BL_x) * dx, (j + BL_y) * dx])
        u1 = interp_u_MAC(vel_x, vel_y, pos, dx)
        # first
        pos1 = pos + neg_dt * u1 * 0.5
        # second
        u2 = interp_u_MAC(vel_x, vel_y, pos1, dx)
        pos2 = pos + neg_dt * u2
        field_new[i, j] = interp_2d(field_old, pos2, dx, BL_x, BL_y)

def fast_marching(phi):
    phi_np = phi.to_numpy().astype(float)
    fm.fm_2d(phi_np, float(narrowband_width), float(dx))
    phi.from_numpy(phi_np)

@ti.kernel
def set_jump_kernel(phi: ti.template(), jump_x: ti.template(), jump_y: ti.template(), dt: float):
    for i, j in jump_x:
        if i > 0 and  i < res_x:
            if phi[i - 1, j] < 0.0 and phi[i, j] >= 0.0:
                theta = get_theta(phi[i - 1, j], phi[i, j])
                pos_left = ti.Vector([(i - 0.5) * dx, (j + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_left + theta * pos_right
                st_jump = sigma * curvature_2d(phi, intf_pos, dx)
                jump_x[i, j] = st_jump * dt / dx
            elif phi[i - 1, j] >= 0.0 and phi[i, j] < 0.0:
                theta = get_theta(phi[i, j], phi[i - 1, j])
                pos_left = ti.Vector([(i - 0.5) * dx, (j + 0.5) * dx])
                pos_right = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_right + theta * pos_left
                st_jump = sigma * curvature_2d(phi, intf_pos, dx)
                jump_x[i, j] = st_jump * dt / dx
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
                jump_y[i, j] = st_jump * dt / dx
            elif phi[i, j - 1] >= 0.0 and phi[i, j] < 0.0:
                theta = get_theta(phi[i, j], phi[i, j - 1])
                pos_bottom = ti.Vector([(i + 0.5) * dx, (j - 0.5) * dx])
                pos_top = ti.Vector([(i + 0.5) * dx, (j + 0.5) * dx])
                intf_pos = (1.0 - theta) * pos_top + theta * pos_bottom
                st_jump = sigma * curvature_2d(phi, intf_pos, dx)
                jump_y[i, j] = st_jump * dt / dx
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

    vel_proj.project(verbose = True)
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
main()