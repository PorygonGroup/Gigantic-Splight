import taichi as ti
import math
from scene import Scene

boundary = (30, 20, 20)

particle_num = 17500 # todo
max_neighbors_num = 3000 # todo
max_particle_num_per_grid = 750 # todo
h = 1.0

neighbor_radius = h * 1.05
cell_size = neighbor_radius * 1.5
grid_size = (int(math.ceil(boundary[0] / cell_size)), int(math.ceil(boundary[1] / cell_size)), int(math.ceil(boundary[2] / cell_size)))
time_delta = 1.0 / 20.0
epsilon = 1e-2
lambda_epsilon = 100.0
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi
mass = 1.0
rho0 = 5.0
corr_deltaQ_coeff = 0.3
corrK = 0.0001
XSPH_c = 0.01
vorti_epsilon = 0.01
g_delta = 0.01

gravity = ti.Vector([0.0, 0.0, -9.8])
force_x_coeff = ti.Vector([1.0, 0.0, 0.0])
force_y_coeff = ti.Vector([0.0, 1.0, 0.0])

'''
Example ParticleSystem Implementation. Particle behaviors should be modified upon future implementation.
'''
@ti.data_oriented
class ParticleSystem:
    def __init__(self, N: int, radius: float, scene: Scene):
        self.N = N
        self.old_p = ti.Vector.field(3, float, shape=N)
        self.p = ti.Vector.field(3, float, shape=N)
        self.v = ti.Vector.field(3, float,shape=N)
        self.f = ti.Vector.field(3, float,shape=N)
        self.XSPH = ti.Vector.field(3, float, shape=N)
        self.lambdas = ti.field(float, N)
        self.delta_p = ti.Vector.field(3, float, N)
        self.radius = radius
        self.scene = scene

        self.particle_neighbors_num = ti.field(int,shape=N)
        self.particle_neighbors = ti.field(int,shape=(N,max_neighbors_num))

        self.grid_particle_num = ti.field(int)
        self.grid_2_particles = ti.field(int)
        self.gNode = ti.root.dense(ti.ijk, grid_size)
        self.gNode.place(self.grid_particle_num)
        self.gNode.dense(ti.l, max_particle_num_per_grid).place(self.grid_2_particles)

        # initial position
        self.init_position()

        # just for renderer
        self.color = ti.Vector.field(3, float, shape=N)


    @ti.func
    def confine_position_to_scene(self, p):
        b_min = self.radius
        b_max = ti.Vector([self.scene.board_states[None][0], boundary[1], boundary[2]]) - self.radius

        collided, new_p = self.scene.collide_with_box(p)
        if collided:
            p = new_p

        for i in ti.static(range(3)):
            if p[i] <= b_min:
                p[i] = b_min + epsilon * ti.random()
            elif b_max[i] <= p[i]:
                p[i] = b_max[i] - epsilon * ti.random()

        return p

    @ti.kernel
    def init_position(self):
        boundary_v = ti.Vector(boundary)
        init_box = ti.Vector([boundary_v[0] * 0.3, boundary_v[1], boundary_v[2] * 0.3])
        offset_box = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.N):
            for c in ti.static(range(3)):
                self.p[i][c] = ti.random(float) * init_box[c] + offset_box[c]
                self.v[i][c] = 0
                self.f[i][c] = 0
        self.scene.init_boarder(boundary)

    @ti.func
    def spiky(self, r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s < h:
            x = (h * h - s * s) / (h * h * h)
            result = poly6_factor * x * x * x
        return result

    @ti.func
    def compute_scorr(self, pos_ji):
        x = self.poly6_value(pos_ji.norm(), h) / self.poly6_value(corr_deltaQ_coeff * h, h)
        x = x * x
        x = x * x
        return -corrK * x

    @ti.func
    def get_grid(self, pos):
        return int(pos // cell_size) 

    @ti.func
    def is_in_grid(self, g):
        return 0 <= g[0] < grid_size[0] and 0 <= g[1] < grid_size[1] and 0<= g[2] < grid_size[2]

    @ti.kernel
    def sub_step(self):
        for p_i in self.p:
            pos_i = self.p[p_i]

            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.particle_neighbors_num[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0: break
                pos_ji = pos_i - self.p[p_j]
                grad_j = self.spiky(pos_ji, h)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                density_constraint += self.poly6_value(pos_ji.norm(), h)

            density_constraint = (mass * density_constraint / rho0) - 1.0
            sum_gradient_sqr += grad_i.dot(grad_i)
            self.lambdas[p_i] = - 1.0 * (density_constraint / (sum_gradient_sqr + lambda_epsilon))

        for p_i in self.p:
            pos_i = self.p[p_i]
            lambda_i = self.lambdas[p_i]

            pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.particle_neighbors_num[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0: break
                lambda_j = self.lambdas[p_j]
                pos_ji = pos_i - self.p[p_j]
                scorr_ij = self.compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * self.spiky(pos_ji, h)

            pos_delta_i /= rho0
            self.delta_p[p_i] = pos_delta_i

        for p_i in self.p:
            self.p[p_i] += self.delta_p[p_i]

    @ti.kernel
    def prologue(self, force_x: float, force_y: float):
        for p_i in self.p:
            self.old_p[p_i] = self.p[p_i]
            self.f[p_i] += force_x * force_x_coeff + force_y * force_y_coeff

        for p_i in self.p:
            self.v[p_i] += (self.f[p_i] + gravity) / mass * time_delta
            self.p[p_i] += self.v[p_i] * time_delta
            self.p[p_i] = self.confine_position_to_scene(self.p[p_i])

        # todo: scene boundary

        for I in ti.grouped(self.grid_particle_num):
            self.grid_particle_num[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        for p_i in self.p:
            cell = self.get_grid(self.p[p_i])
            offset = ti.atomic_add(self.grid_particle_num[cell], 1)
            self.grid_2_particles[cell, offset] = p_i

        for p_i in self.p:
            pos_i = self.p[p_i]
            grid = self.get_grid(pos_i)
            neighbor_num = 0
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                grid_ = grid + offset
                if self.is_in_grid(grid_):
                    for j in range(self.grid_particle_num[grid_]):
                        p_j = self.grid_2_particles[grid_, j]
                        if neighbor_num < max_neighbors_num and p_j != p_i and (pos_i - self.p[p_j]).norm() < neighbor_radius:
                            self.particle_neighbors[p_i, neighbor_num] = p_j
                            neighbor_num += 1
            self.particle_neighbors_num[p_i] = neighbor_num

    @ti.kernel
    def epilogue(self):
        for p_i in self.p:
            pos = self.p[p_i]
            self.p[p_i] = self.confine_position_to_scene(pos)
            self.v[p_i] = (self.p[p_i] - self.old_p[p_i]) / time_delta

        for p_i in self.p:
            pos_i = self.p[p_i]

            omega_i = ti.Vector([0.0, 0.0, 0.0])
            XSPH_i = ti.Vector([0.0, 0.0, 0.0])

            dx_i = ti.Vector([0.0, 0.0, 0.0])
            dy_i = ti.Vector([0.0, 0.0, 0.0])
            dz_i = ti.Vector([0.0, 0.0, 0.0])
            n_dx_i = ti.Vector([0.0, 0.0, 0.0])
            n_dy_i = ti.Vector([0.0, 0.0, 0.0])
            n_dz_i = ti.Vector([0.0, 0.0, 0.0])
            dx = ti.Vector([g_delta, 0.0, 0.0]) 
            dy = ti.Vector([0.0, g_delta, 0.0])
            dz = ti.Vector([0.0, 0.0, g_delta])

            for j in range(self.particle_neighbors_num[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0: break
                pos_ji = pos_i - self.p[p_j]
                vel_ij = self.v[p_j] - self.v[p_i]
                omega_i += vel_ij.cross(self.spiky(pos_ji, h))
                dx_i += vel_ij.cross(self.spiky(pos_ji + dx, h))
                dy_i += vel_ij.cross(self.spiky(pos_ji + dy, h))
                dz_i += vel_ij.cross(self.spiky(pos_ji + dz, h))
                n_dx_i += vel_ij.cross(self.spiky(pos_ji - dx, h))
                n_dy_i += vel_ij.cross(self.spiky(pos_ji - dy, h))
                n_dz_i += vel_ij.cross(self.spiky(pos_ji - dz, h))
                XSPH_i += self.poly6_value(pos_ji.norm(), h) * vel_ij

            n_x = (dx_i.norm() - n_dx_i.norm()) / (2 * g_delta)
            n_y = (dy_i.norm() - n_dy_i.norm()) / (2 * g_delta)
            n_z = (dz_i.norm() - n_dz_i.norm()) / (2 * g_delta)
            n = ti.Vector([n_x, n_y, n_z]).normalized()
            self.f[p_i] = vorti_epsilon * n.cross(omega_i) if not omega_i.norm() == 0.0 else 0.0
            self.XSPH[p_i] = XSPH_i * XSPH_c
        for p_i in self.p:
            self.v[p_i] += self.XSPH[p_i]

    @ti.kernel
    def recolor_debug_ver(self):
        for v_i in self.delta_p:
            self.color[v_i][2] = 0
            if self.v[v_i][0] > 0.0:
                self.color[v_i][0] = self.v[v_i][0]
            else:
                self.color[v_i][2] = -self.v[v_i][0]

    @ti.kernel
    def recolor(self):
        for v_i in self.v:
            self.color[v_i][2] = 240/255
            t = self.v[v_i].norm()/20
            if t>1:t=1
            R_BASE = 37/255
            G_BASE = 157/255
            self.color[v_i][0] = R_BASE + (2*R_BASE-R_BASE)*t
            self.color[v_i][1] = G_BASE + (1.3*G_BASE-G_BASE)*t
