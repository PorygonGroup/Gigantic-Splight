import taichi as ti
import math

cell_size = 1 / 16
'''
TODO: customize our particle arrangement
'''


@ti.data_oriented
class ParticleSystem:
    def __init__(self, N: int, radius: float):
        self.N = N
        self.x = ti.Vector.field(3, float, (N, N))
        self.v = ti.Vector.field(3, float, (N, N))
        num_triangles = (N - 1) * (N - 1) * 2
        self.indices = ti.field(int, num_triangles * 3)
        self.vertices = ti.Vector.field(3, float, N * N)
        self.radius = radius

        # TODO - modify the following!
        for i, j in ti.ndrange(N, N):
            self.x[i, j] = ti.Vector([i * cell_size, j * cell_size, ti.sin((i + j)) / 4])
        self.set_indices()
        self.set_vertices()

    @ti.kernel
    def step(self):
        pass

    @ti.kernel
    def set_indices(self):
        N = self.N
        for i, j in ti.ndrange(N, N):
            if i < N - 1 and j < N - 1:
                square_id = (i * (N - 1)) + j
                # 1st triangle of the square
                self.indices[square_id * 6 + 0] = i * N + j
                self.indices[square_id * 6 + 1] = (i + 1) * N + j
                self.indices[square_id * 6 + 2] = i * N + (j + 1)
                # 2nd triangle of the square
                self.indices[square_id * 6 + 3] = (i + 1) * N + j + 1
                self.indices[square_id * 6 + 4] = i * N + (j + 1)
                self.indices[square_id * 6 + 5] = (i + 1) * N + j

    @ti.kernel
    def set_vertices(self):
        N = self.N
        for i, j in ti.ndrange(N, N):
            self.vertices[i * N + j] = self.x[i, j]
