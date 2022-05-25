import taichi as ti
from taichi import Field
import numpy as np
from pbf3d import ParticleSystem

class Renderer:

    def __init__(self, part_sys: ParticleSystem):
        self.part_sys = part_sys
        self.window = ti.ui.Window("Render particles", (800, 800), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        camera = ti.ui.make_camera()
        camera.position(0, 0.5, 2)
        camera.lookat(1, 0, 0)
        camera.fov(90)
        self.scene.set_camera(camera)
        self.camera = camera

    def render(self):
        while self.window.running:
            self.part_sys.step()
            scene = self.scene
            scene.point_light(pos=(0.0, 0.5, 2), color=(1, 1, 1))
            scene.mesh(self.part_sys.vertices, indices=self.part_sys.indices, color=(0.5, 0.5, 0.5), two_sided=True)
            self.canvas.scene(scene)
            self.window.show()