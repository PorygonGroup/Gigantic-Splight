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
        self.camera_pos = np.array([0, 0.5, 2], dtype=np.float64)
        self.camera_dir = np.array([0, 0, -1], dtype=np.float64)
        camera = ti.ui.make_camera()
        camera.position(0, 0.5, 2)
        camera.lookat(1, 0, 0)
        camera.fov(90)
        self.scene.set_camera(camera)
        self.camera = camera

    def updateCamera(self):
        # TODO: make movement more intuitive
        self.window.get_event()
        POS_EPS = 0.01
        pos_delta = np.array([0.0, 0.0, 0.0])
        if self.window.is_pressed('w'):
            pos_delta[0] += POS_EPS
        if self.window.is_pressed('s'):
            pos_delta[0] -= POS_EPS
        if self.window.is_pressed('a'):
            pos_delta[1] += POS_EPS
        if self.window.is_pressed('d'):
            pos_delta[1] -= POS_EPS
        if self.window.is_pressed('q'):
            pos_delta[2] += POS_EPS
        if self.window.is_pressed('e'):
            pos_delta[2] -= POS_EPS

        if self.window.is_pressed(ti.GUI.SHIFT):
            self.camera_pos += pos_delta
        else:
            self.camera_dir += pos_delta
        self.camera.position(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
        self.camera.lookat(self.camera_dir[0], self.camera_dir[1], self.camera_dir[2])
        self.camera.fov(90)
        self.scene.set_camera(self.camera)

    def render(self):
        while self.window.running:
            self.updateCamera()
            self.part_sys.step()
            self.scene.point_light(pos=(0.0, 0.5, 2), color=(1, 1, 1))
            # TODO: maybe other options are better
            self.scene.mesh(self.part_sys.vertices, indices=self.part_sys.indices, color=(0.5, 0.5, 0.5), two_sided=True)
            self.canvas.scene(self.scene)
            self.window.show()
