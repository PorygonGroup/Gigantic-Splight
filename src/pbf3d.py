import taichi as ti
import math

time_delta = 1.0 / 20.0
epsilon = 1e-5
lambda_epsilon = 100.0
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi
h = 1.1
mass = 1.0
rho0 = 1.0
corr_deltaQ_coeff = 0.3
corrK = 0.001
solverIterations = 5
XSPH_c = 0.01

'''
TODO: customize our particle arrangement
'''

'''
Example ParticleSystem Implementation. Particle behaviors should be modified upon future implementation.
'''
@ti.data_oriented
class ParticleSystem:
    def __init__(self, N: int, radius: float):
        self.N = N
        self.p = ti.Vector.field(3, float, N)
        self.v = ti.Vector.field(3, float, N)
        self.f = ti.Vector.field(3, float, N)
        self.radius = radius

        self.lambdas = ti.field(float, N)
        self.delta_p = ti.Vector.field(3, float, N)
        self.delta_v = ti.Vector.field(3, float, N)
        self.omega = ti.Vector.field(3, float, N)
        self.particle_num_neighbors = ti.field(int)

        # to do: initial position

    @ti.func
    def spiky(self, r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h);
            result = poly6_factor * x * x * x
        return result

    @ti.func
    def compute_scorr(self, pos_ji):
        x = self.poly6_value(pos_ji.norm(), h) / self.poly6_value(corr_deltaQ_coeff * h, h)
        x = x * x
        x = x * x
        return -corrK * x


    @ti.kernel
    def sub_step(self):
        for p_i in self.p:
            pos_i = self.p[p_i]

            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.paritcle_neighbors_num[p_i]):
                p_j = particle_neighbors[p_i, j]
                if p_j < 0: break
                pos_ji = pos_i - self.p[p_j]
                grad_j = self.spiky(pos_ji, h)
                sum_gradient_sqr += grad_j.dot(grad_j)
                density_constraint += self.poly6_value(pos_ji.norm(), h)

            density_constraint = (mass * density_constraint / rho0) - 1.0
            sum_gradient_sqr += grad_i.dot(grad_i)
            self.lambdas[p_i] = - density_constraint / (sum_gradient_sqr + lambda_epsilon)

        for p_i in self.p:
            pos_i = self.p[p_i]
            lambda_i = self.lambdas[p_i]

            pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0: break
                lambda_j = self.lambdas[p_j]
                pos_ji = pos_i = self.p[p_j]
                scorr_ij = self.compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * self.spiky(pos_ji, h)

            pos_delta_i /= rho0
            self.delta_p[p_i] = pos_delta_i

        for p_i in self.p:
            self.p[p_i] += self.delta_p[p_i]

    @ti.kernel
    def prologue(self):
        # to do: add gravity

        # to do: scene boundary

        # to do: compute neighbors
        pass

    @ti.kernel
    def epilogue(self):
        for p_i in self.p:
            pos = self.p[p_i]
            self.p[p_i] = confine_position_to_scene(pos)

        for p_i in self.p:
            pos_i = self.p[p_i]

            omega_i = ti.Vector([0.0, 0.0, 0.0])

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.paritcle_neighbors[p_i, j]
                if p_j < 0: break
                pos_ji = pos_i - self.p[p_j]
                vel_ij = self.v[p_j] - self.v[p_i]
                omega_i += vel_ij.cross(self.spiky(pos_ji, h))

            self.omega[p_i] = omega_i

class Simulator:
    def __init__(self, part_sys: ParticleSystem):
        self.part_sys = part_sys

    def step(self):
        self.part_sys.prologue()
        for _ in range(solverIterations):
            self.part_sys.sub_step()
        self.part_sys.epilogue()

