import taichi as ti
import numpy as np


# @ti.data_oriented
# class Vec3:
#     def __init__(self, x: float, y: float, z: float) -> None:
#         self.value = ti.Vector([x, y, z])
#         # self.value[0] = x
#         # self.value[1] = y
#         # self.value[2] = z
#         self.x = x
#         self.y = y
#         self.z = z

#     def norm(self) -> float:
#         return (self.x * self.x + self.y * self.y + self.z * self.z) ** (0.5)

#     def normalize(self):
#         len = self.norm()
#         return Vec3(self.x / len, self.y / len, self.z / len)

#     def __add__(self, operand):
#         return Vec3(self.x + operand.x, self.y + operand.y, self.z + operand.z)

#     def __mul__(self, num):
#         return Vec3(self.x * num, self.y * num, self.z * num)

#     def __rmul__(self, num):
#         return self.__mul__(num)

#     def __getitem__(self, key):
#         return self.value[key]

#     def __setitem__(self, key, value):
#         self.value[key] = value
#         if key == 0:
#             self.x = value
#         elif key == 1:
#             self.y = value
#         else:
#             self.z = value

#     @staticmethod
#     def dot(a, b):
#         return a.x * b.x + a.y * b.y + a.z * b.z

#     @staticmethod
#     def cross(a, b):
#         return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


@ti.data_oriented
class Box:
    # def __init__(self, p0, u_dir_, len_a, len_b, len_c) -> None:
    #     self.p0 = p0
    #     u_dir = u_dir_.normalized()
    #     assert u_dir[2] == 0.0
    #     # vectors of edges
    #     self.e = ti.Vector.field(3, float, shape=3)
    #     self.e[0] = u_dir * len_a
    #     self.e[1] = (ti.Vector([-u_dir.y, u_dir.x, 0]) * len_b,)
    #     self.e[2] = (ti.Vector([0.0, 0.0, len_c]),)
    #     self.get_dir_len()

    def __init__(self, box):
        self.p0 = ti.Vector([box.n0[0], box.n0[1], 0.0])
        self.e = ti.Vector.field(3, float, shape=3)
        self.e[0] = ti.Vector([box.n1[0] - box.n0[0], box.n1[1] - box.n0[1], 0.])
        self.e[1] = ti.Vector([box.n3[0] - box.n0[0], box.n3[1] - box.n0[1], 0.])
        self.e[2] = ti.Vector([0.0, 0.0, box.h])
        print(self.p0, self.e)
        self.get_dir_len()

    def get_dir_len(self):
        self.e_dir = ti.Vector.field(3, float, shape=3)
        self.len = ti.Vector([0., 0., 0.])
        for i in range(3):
            self.e_dir[i] = self.e[i].normalized()
            self.len[i] = self.e[i].norm()
        print(self.len)

    @ti.func
    def collide(self, p, epsilon):
        o = p - self.p0

        # in the cuboid space : Position in the Box
        pb = ti.Vector([0., 0., 0.])
        pb[0] = o.dot(self.e_dir[0])
        pb[1] = o.dot(self.e_dir[1])
        pb[2] = o.z
        ret = ti.Vector([0., 0., 0.])
        collided = False
        if (
            0 < pb[0] < self.len[0]
            and 0 < pb[1] < self.len[1]
            and 0 < pb[2] < self.len[2]
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

            # axis_max = -1.0
            # which_max = -1
            # for i in ti.static(range(3)):
            #     if pb[i] > axis_max:
            #         axis_max = pb[i]
            #         which_max = i
            #     if pb[i] < axis_min:
            #         axis_min = pb[i]
            #         which_min = i

            # print(axis_min, which_min, axis_max, which_max)

            # update the new position in cuboid space
            for i in ti.static(range(3)):
                if which_min == i:
                    if is_lm:
                        pb[i] = self.len[i]
                    else:
                        pb[i] = 0.0
                    break

            collided = True
            x_dir = ti.Vector([self.e_dir[0][0], -self.e_dir[0][1], 0.])
            y_dir = ti.Vector([self.e_dir[0][1], self.e_dir[0][0], 0.])

            ret[0] = pb.dot(x_dir)
            ret[1] = pb.dot(y_dir)
            ret[2] = pb[2]
            # ret[0] = p[0] + 1
            # ret[1] = p[1]
            # ret[2] = p[2]

        return collided, ret


@ti.data_oriented
class Scene:
    def __init__(self, box):
        self.board_states = ti.Vector.field(2, float, shape=())
        # boxes : 4 points on the ground + height.
        self.box = Box(box)
        self.time_delta = 1.0 / 20.0
        self.epsilon = 0.01

    def update(self):
        # self.update_board()
        pass
        # self.update_box()

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
        return self.box.collide(p, self.epsilon)
