import taichi
import taichi as ti
from taichi import Field
import numpy as np
from pbf3d import ParticleSystem
from scene import Scene

INIT_CAMERA_POS = np.array([0.5, -2, 1.3], dtype=np.float64)
INIT_CAMERA_DIR = np.array([0.5, 1, 0], dtype=np.float64)

'''
Update a Cartesian coordinate system with theta and phi angles.
'''


def updateCartCoorByAngle(cartCoor, vert, hori):
    norm = np.linalg.norm(cartCoor)
    phi = np.arctan2(cartCoor[1], cartCoor[0])
    theta = np.arccos(cartCoor[2] / norm)
    phi += hori
    theta += vert
    # theta = np.clip(theta, -np.pi/2, np.pi/2)

    cartCoor[0] = np.cos(phi) * np.sin(theta)
    cartCoor[1] = np.sin(phi) * np.sin(theta)
    cartCoor[2] = np.cos(theta)


class Renderer:

    def __init__(self, part_sys: ParticleSystem, scene_info: Scene):
        self.part_sys = part_sys
        self.scene_info = scene_info
        self.window = ti.ui.Window("Render particles", (800, 800), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera_pos = INIT_CAMERA_POS.copy()
        self.camera_dir = INIT_CAMERA_DIR.copy()
        camera = ti.ui.make_camera()
        self.camera = camera
        self.boxes = []
        self.updateCamera()

    def updateCamera(self):
        self.window.get_event()
        POS_EPS = 0.02
        DIR_EPS = 0.01
        pos_delta = np.array([0.0, 0.0, 0.0])
        vert_dir_delta = 0.0
        hori_dir_delta = 0.0
        if self.window.is_pressed('w'):
            pos_delta[0] += POS_EPS
        if self.window.is_pressed('s'):
            pos_delta[0] -= POS_EPS
        if self.window.is_pressed('a'):
            pos_delta[1] -= POS_EPS
        if self.window.is_pressed('d'):
            pos_delta[1] += POS_EPS
        if self.window.is_pressed(' '):
            pos_delta[2] += POS_EPS
        if self.window.is_pressed(ti.GUI.SHIFT):
            pos_delta[2] -= POS_EPS
        if self.window.is_pressed(ti.GUI.UP):
            vert_dir_delta -= DIR_EPS
        if self.window.is_pressed(ti.GUI.DOWN):
            vert_dir_delta += DIR_EPS
        if self.window.is_pressed(ti.GUI.LEFT):
            hori_dir_delta += DIR_EPS
        if self.window.is_pressed(ti.GUI.RIGHT):
            hori_dir_delta -= DIR_EPS

        if self.window.is_pressed('r'):
            self.camera_pos = INIT_CAMERA_POS.copy()
            self.camera_dir = INIT_CAMERA_DIR.copy()

        view = self.camera_dir - self.camera_pos
        # Update camera position
        move_x = np.array([view[0], view[1], 0])
        move_x /= np.linalg.norm(move_x)
        move_y = np.array([view[1], -view[0], 0])
        move_y /= np.linalg.norm(move_y)
        move_delta = pos_delta[0] * move_x + pos_delta[1] * move_y + np.array([0, 0, pos_delta[2]])
        self.camera_pos += move_delta
        self.camera_dir += move_delta

        # Update camera direction
        updateCartCoorByAngle(view, vert_dir_delta, hori_dir_delta)
        self.camera_dir = view + self.camera_pos

        # Update camera
        self.camera.position(*self.camera_pos)
        self.camera.lookat(*self.camera_dir)
        self.camera.up(*np.array([0, 0, 1]))
        self.camera.fov(90)
        self.scene.set_camera(self.camera)

    def addBox(self,box):
        class Object(object):
            pass
        boxObj = Object()
        boxObj.vert = taichi.Vector.field(3,float,shape = 8)
        for i,h in enumerate([0,box.h]):
            for j,n in enumerate([box.n0,box.n1,box.n2,box.n3]):
                boxObj.vert[i*4+j] = ti.Vector([n[0],n[1],h])
        boxObj.idx = taichi.field(int,6*6)
        cor = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        for i,st in enumerate(cor):
            # first triangle
            boxObj.idx[i*6] = st[0]
            boxObj.idx[i*6+1] = st[1]
            boxObj.idx[i*6+2] = st[3]
            # second triangle
            boxObj.idx[i*6+3] = st[1]
            boxObj.idx[i*6+4] = st[2]
            boxObj.idx[i*6+5] = st[3]
        boxObj.color = (0.2,0.6,0.2)
        self.boxes.append(boxObj)


    def render(self):
        if self.window.running:
            self.updateCamera()
            # TODO: maybe other options are better
            scene = self.scene
            scene.ambient_light((0.8, 0.2, 0.2))
            # scene.point_light(pos=(2.0, 0.5, 1), color=(0.7, 0.3, 0))
            scene.point_light(pos=(0.0, 0.5, 2), color=(1, 1, 1))
            scene.particles(self.part_sys.p, self.part_sys.radius)
            for b in self.boxes:
                scene.mesh(b.vert,indices=b.idx,color=b.color,two_sided=True)
            self.canvas.scene(scene)
            self.canvas.set_background_color((0.6, 0.6, 0.6))
            self.window.show()
