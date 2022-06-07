import taichi as ti

@ti.data_oriented
class Scene:
    def __init__(self):
        self.board_states = ti.Vector.field(3, float)
