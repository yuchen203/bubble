import taichi as ti
from solver import *
from utils import *


# assume the domain is surrounded by walls
# assume rho_ratio = rho_L / rho_A > 1
# assume phi < 0 indicate the liquid region
@ti.data_oriented
class TwoPhaseVelProjJump2d:
    def __init__(self, res, dx, phi, vel, rho_L, rho_G, base_level=3, real=float):
        self.res = res
        self.dx = dx
        self.phi = phi
        self.vel = vel
        self.rho_L = rho_L
        self.rho_G = rho_G
        self.real = real

        self.jump = [
            ti.field(dtype=real, shape=(res[0] + 1, res[1])),
            ti.field(dtype=real, shape=(res[0], res[1] + 1)),
        ]
        self.solver = AMGPCG_2D(res, base_level, real)

    def project(
        self, dt, max_iters=-1, verbose=False, rel_tol=1e-12, abs_tol=1e-14, eps=1e-20
    ):
        self.set_bc_kernel()
        self.calc_div_kernel(self.solver.b)
        self.add_jump_to_rhs_kernel(self.solver.b, dt)
        self.set_lhs_kernel(
            self.solver.is_dof[0],
            self.solver.Adiag[0],
            self.solver.Ax[0],
            dt
        )
        self.solver.build()
        self.solver.solve(True, max_iters, verbose, rel_tol, abs_tol, eps)
        self.apply_pressure_kernel(self.solver.x, self.solver.Ax[0])

    @ti.kernel
    def set_bc_kernel(self):
        for i, j in self.vel[0]:
            if i == 0 or i == self.res[0]:
                self.vel[0][i, j] = 0.0
        for i, j in self.vel[1]:
            if j == 0 or j == self.res[1]:
                self.vel[1][i, j] = 0.0

    @ti.kernel
    def calc_div_kernel(self, div: ti.template()):
        for i, j in div:
            div[i, j] = (
                self.vel[0][i, j]
                - self.vel[0][i + 1, j]
                + self.vel[1][i, j]
                - self.vel[1][i, j + 1]
            ) / self.dx

    @ti.kernel
    def add_jump_to_rhs_kernel(self, div: ti.template(), dt: float):
        for i, j in self.phi:
            if self.phi[i, j] < 0.0:
                if i > 0 and self.phi[i - 1, j] >= 0.0:
                    theta = get_theta(self.phi[i - 1, j], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] += self.jump[0][i, j] / rho * dt / (self.dx * self.dx)
                if i < self.res[0] - 1 and self.phi[i + 1, j] >= 0.0:
                    theta = get_theta(self.phi[i + 1, j], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] += self.jump[0][i + 1, j] / rho * dt / (self.dx * self.dx)
                if j > 0 and self.phi[i, j - 1] >= 0.0:
                    theta = get_theta(self.phi[i, j - 1], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] += self.jump[1][i, j] / rho * dt / (self.dx * self.dx)
                if j < self.res[1] - 1 and self.phi[i, j + 1] >= 0.0:
                    theta = get_theta(self.phi[i, j + 1], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] += self.jump[1][i, j + 1] / rho * dt / (self.dx * self.dx)
            else:
                if i > 0 and self.phi[i - 1, j] < 0.0:
                    theta = get_theta(self.phi[i - 1, j], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] -= self.jump[0][i, j] / rho * dt / (self.dx * self.dx)
                if i < self.res[0] - 1 and self.phi[i + 1, j] < 0.0:
                    theta = get_theta(self.phi[i + 1, j], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] -= self.jump[0][i + 1, j] / rho * dt / (self.dx * self.dx)
                if j > 0 and self.phi[i, j - 1] < 0.0:
                    theta = get_theta(self.phi[i, j - 1], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] -= self.jump[1][i, j] / rho * dt / (self.dx * self.dx)
                if j < self.res[1] - 1 and self.phi[i, j + 1] < 0.0:
                    theta = get_theta(self.phi[i, j + 1], self.phi[i, j])
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    div[i, j] -= self.jump[1][i, j + 1] / rho * dt / (self.dx * self.dx)

    @ti.kernel
    def set_lhs_kernel(
        self, is_dof: ti.template(), Adiag: ti.template(), Ax: ti.template(), dt: float
    ):
        for i, j in is_dof:
            is_dof[i, j] = True

        for i, j in Ax:
            if i < self.res[0] - 1:
                phi0 = self.phi[i, j]
                phi1 = self.phi[i + 1, j]
                if phi0 < 0.0 and phi1 < 0.0:
                    Ax[i, j][0] = -1.0 / self.rho_L * dt / (self.dx * self.dx)
                elif phi0 < 0.0 and phi1 >= 0.0:
                    theta = get_theta(phi0, phi1)
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    Ax[i, j][0] = -1.0 / rho * dt / (self.dx * self.dx)
                elif phi0 >= 0.0 and phi1 < 0.0:
                    theta = get_theta(phi1, phi0)
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    Ax[i, j][0] = -1.0 / rho * dt / (self.dx * self.dx)
                else:
                    Ax[i, j][0] = -1.0 / self.rho_G * dt / (self.dx * self.dx)
            else:
                Ax[i, j][0] = 0.0

            if j < self.res[1] - 1:
                phi0 = self.phi[i, j]
                phi1 = self.phi[i, j + 1]
                if phi0 < 0.0 and phi1 < 0.0:
                    Ax[i, j][1] = -1.0 / self.rho_L * dt / (self.dx * self.dx)
                elif phi0 < 0.0 and phi1 >= 0.0:
                    theta = get_theta(phi0, phi1)
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    Ax[i, j][1] = -1.0 / rho * dt / (self.dx * self.dx)
                elif phi0 >= 0.0 and phi1 < 0.0:
                    theta = get_theta(phi0, phi1)
                    rho = theta * self.rho_L + (1.0 - theta) * self.rho_G
                    Ax[i, j][1] = -1.0 / rho * dt / (self.dx * self.dx)
                else:
                    Ax[i, j][1] = -1.0 * dt / (self.dx * self.dx)
            else:
                Ax[i, j][1] = 0.0

        for i, j in Adiag:
            ret = 0.0
            if i > 0:
                ret -= Ax[i - 1, j][0]
            if i < self.res[0] - 1:
                ret -= Ax[i, j][0]
            if j > 0:
                ret -= Ax[i, j - 1][1]
            if j < self.res[1] - 1:
                ret -= Ax[i, j][1]
            Adiag[i, j] = ret

    @ti.kernel
    def apply_pressure_kernel(self, pressure: ti.template(), Ax: ti.template()):
        for i, j in self.vel[0]:
            if i > 0 and i < self.res[0]:
                if self.phi[i, j] < 0.0 and self.phi[i - 1, j] < 0.0:
                    p0 = pressure[i - 1, j]
                    p1 = pressure[i, j]
                    self.vel[0][i, j] += (p1 - p0) * Ax[i - 1, j][0] * self.dx
                elif self.phi[i, j] >= 0.0 and self.phi[i - 1, j] >= 0.0:
                    p0 = pressure[i - 1, j]
                    p1 = pressure[i, j]
                    self.vel[0][i, j] += (p1 - p0) * Ax[i - 1, j][0] * self.dx
                elif self.phi[i, j] < 0.0 and self.phi[i - 1, j] >= 0.0:
                    p0 = pressure[i - 1, j] + self.jump[0][i, j]
                    p1 = pressure[i, j]
                    self.vel[0][i, j] += (p1 - p0) * Ax[i - 1, j][0] * self.dx
                else:
                    p0 = pressure[i - 1, j]
                    p1 = pressure[i, j] + self.jump[0][i, j]
                    self.vel[0][i, j] += (p1 - p0) * Ax[i - 1, j][0] * self.dx
        for i, j in self.vel[1]:
            if j > 0 and j < self.res[1]:
                if self.phi[i, j] < 0.0 and self.phi[i, j - 1] < 0.0:
                    p0 = pressure[i, j - 1]
                    p1 = pressure[i, j]
                    self.vel[1][i, j] += (p1 - p0) * Ax[i, j - 1][1] * self.dx
                elif self.phi[i, j] >= 0.0 and self.phi[i, j - 1] >= 0.0:
                    p0 = pressure[i, j - 1]
                    p1 = pressure[i, j]
                    self.vel[1][i, j] += (p1 - p0) * Ax[i, j - 1][1] * self.dx
                elif self.phi[i, j] < 0.0 and self.phi[i, j - 1] >= 0.0:
                    p0 = pressure[i, j - 1] + self.jump[1][i, j]
                    p1 = pressure[i, j]
                    self.vel[1][i, j] += (p1 - p0) * Ax[i, j - 1][1] * self.dx
                else:
                    p0 = pressure[i, j - 1]
                    p1 = pressure[i, j] + self.jump[1][i, j]
                    self.vel[1][i, j] += (p1 - p0) * Ax[i, j - 1][1] * self.dx
