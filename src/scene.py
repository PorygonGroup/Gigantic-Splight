import taichi as ti
import numpy as np


@ti.data_oriented
class Box:

    def __init__(self, box):
        self.p0 = ti.Vector([box.n0[0], box.n0[1], 0.0])
        self.e = ti.Vector.field(3, float, shape=3)
        self.e[0] = ti.Vector([box.n1[0] - box.n0[0], box.n1[1] - box.n0[1], 0.])
        self.e[1] = ti.Vector([box.n3[0] - box.n0[0], box.n3[1] - box.n0[1], 0.])
        # swap e_0 and e_1 when e_0 cross e_1 is not in +z direction
        if self.e[0].cross(self.e[1]).z < 0:
            self.e[0], self.e[1] = self.e[1], self.e[0]
        self.e[2] = ti.Vector([0.0, 0.0, box.h])
        self.get_dir_len()
        self.world2box, self.box2world = self.getTransMatrices()

    def getTransMatrices(self):
        new_origin = self.p0
        cos_theta = self.e_dir[0][0]
        sin_theta = self.e_dir[0][1]
        T = np.array([[1, 0, 0, -new_origin[0]], [0, 1, 0, -new_origin[1]], [0, 0, 1, -new_origin[2]], [0, 0, 0, 1]])
        R = np.array([[cos_theta, sin_theta, 0, 0], [-sin_theta, cos_theta, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        pos = R @ T
        inv = np.linalg.inv(pos)
        return ti.Matrix(pos), ti.Matrix(inv)

    def get_dir_len(self):
        self.e_dir = ti.Vector.field(3, float, shape=3)
        self.len = ti.Vector([0., 0., 0.])
        for i in range(3):
            self.e_dir[i] = self.e[i].normalized()
            self.len[i] = self.e[i].norm()
        print(self.len)

    @ti.func
    def collide(self, p, radius, epsilon):

        # in the cuboid space : Position in the Box
        pb_inter = self.world2box @ ti.Vector([p[0], p[1], p[2], 1.0])
        pb = ti.Vector([pb_inter[0], pb_inter[1], pb_inter[2]])
        ret = ti.Vector([0., 0., 0.])
        collided = False

        if (
                -radius < pb[0] < self.len[0] + radius
                and -radius < pb[1] < self.len[1] + radius
                and -radius < pb[2] < self.len[2] + radius
        ):
            # FIXME: may have precision issues
            # return the projection point of the closest surface

            # len minus pb
            pb_lm = self.len - pb

            axis_min = 1e50
            which_min = -1
            is_lm = False
            for i in ti.static(range(3)):
                if pb[i] < axis_min:
                    axis_min = pb[i]
                    which_min = i
                    is_lm = False
                if pb_lm[i] < axis_min:
                    axis_min = pb_lm[i]
                    which_min = i
                    is_lm = True
            assert which_min != -1

            # update the new position in cuboid space
            for i in ti.static(range(3)):
                if which_min == i:
                    if is_lm:
                        pb[i] = self.len[i] + radius
                    else:
                        pb[i] = -radius

            collided = True
            ret_inter = self.box2world @ ti.Vector([pb[0], pb[1], pb[2], 1.0])
            ret = ti.Vector([ret_inter[0], ret_inter[1], ret_inter[2]])

        return collided, ret


@ti.data_oriented
class Scene:
    def __init__(self, radius, box=None):
        self.board_states = ti.Vector.field(2, float, shape=())
        # boxes : 4 points on the ground + height.
        self.box = None if box is None else Box(box)
        self.time_delta = 1.0 / 20.0
        self.epsilon = 1e-2  # todo: make this same to that in pbf3d.py
        self.radius = radius
        self.enableBoard = False

    def update(self):
        if self.enableBoard:
            self.update_board()

    def toggleBoard(self, enable=None):
        if enable is None:
            self.enableBoard = not self.enableBoard
        else:
            self.enableBoard = enable

    def update_board(self):
        # move board
        b = self.board_states[None]
        period = 90
        vel_strength = 8.0
        b[1] += 1.0
        if b[1] >= 2 * period:
            b[1] = 0
        b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * self.time_delta
        self.board_states[None] = b

    @ti.func
    def init_boarder(self, boundary):
        self.board_states[None] = ti.Vector([boundary[0] - self.epsilon, -0.0])

    @ti.func
    def collide_with_box(self, p):
        collided, ret = self.box.collide(p, self.radius, self.epsilon)
        return collided, ret
