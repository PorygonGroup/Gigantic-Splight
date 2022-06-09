import taichi as ti
import numpy as np


class Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def norm(self) -> float:
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** (0.5)

    def normalize(self):
        len = self.norm()
        return Vec3(self.x / len, self.y / len, self.z / len)

    def __add__(self, operand):
        return Vec3(self.x + operand.x, self.y + operand.y, self.z + operand.z)

    def __mul__(self, num):
        return Vec3(self.x * num, self.y * num, self.z * num)

    def __rmul__(self, num):
        return self.__mul__(num)

    @staticmethod
    def dot(a, b):
        return a.x * b.x + a.y * b.y + a.z * b.z

    @staticmethod
    def cross(a, b):
        return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


@ti.data_oriented
class Box:
    def __init__(self, p0: Vec3, u_dir_: Vec3, len_a, len_b, len_c) -> None:
        self.p0 = p0
        u_dir = u_dir_.normalize()
        assert u_dir.z == 0.0
        self.u = u_dir * len_a
        self.v = Vec3(-u_dir.y, u_dir.x, 0) * len_b
        self.w = Vec3(0.0, 0.0, 1.0) * len_c  # must be perpendicular to the xy surface

    def collide(self, p):
        o = p - self.p0
        dot_u = Vec3.dot(o, self.u)
        dot_v = Vec3.dot(o, self.v)
        dot_w = Vec3.dot(o, self.w)
        if (
            0 < dot_u < Vec3.dot(self.u, self.u)
            and 0 < dot_v < Vec3.dot(self.v, self.v)
            and 0 < dot_w < Vec3.dot(self.w, self.w)
        ):
            # FIXME: may have precision issues
            return True
        else:
            return False


@ti.data_oriented
class Scene:
    def __init__(self):
        self.board_states = ti.Vector.field(3, float, shape=())
        # boxes : 4 points on the ground + height.
        self.boxes = ti.Vector.field(3, float, shape=())

    def update(self):
        pass

    @ti.func
    def init_boarder(self, boundary):
        self.board_states[None] = ti.Vector([boundary[0], boundary[1], boundary[2]])
