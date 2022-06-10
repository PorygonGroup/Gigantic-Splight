import time

import taichi as ti

from scene import Scene
from pbf3d import ParticleSystem
import pbf3d
from simulator import Simulator

# ti.init(arch=ti.gpu)

arch = ti.vulkan
ti.init(arch=arch)

screen_res = (1920, 1080)
radius = 0.2
fps = 60


class Object(object):
    pass


def getBox(ls, h):
    obj = Object()
    obj.n0 = ls[0]
    obj.n1 = ls[1]
    obj.n2 = ls[2]
    obj.n3 = ls[3]
    obj.h = h
    return obj


if __name__ == '__main__':
    fps_correspond_time = 1.0 / fps
    box = getBox([(29, 8), (35, 6), (37, 12), (31, 14)], 5.2)
    gx, gy = pbf3d.boundary[0], pbf3d.boundary[1]
    ground = getBox([(0.0, 0.0), (0.0, gy), (gx, gy), (gx, 0)], 0.001)
    scene = Scene(box=box)
    ps = ParticleSystem(pbf3d.particle_num, radius, scene)
    rd = Simulator(ps, scene, enableBall=True)
    rd.addBox(box, (0.3, 0.4, 0.5))
    rd.addBox(ground, (0, 0, 0))
    for point in [(0, 0), (0, gy), (gx, gy), (gx, 0)]:
        D = 0.05
        rd.addBox(getBox([(point[0] - D, point[1] - D), (point[0] - D, point[1] + D), (point[0] + D, point[1] + D),
                          (point[0] + D, point[1] - D)], 7), (0.8, 0.4, 0.2))
    bit = 100000
    last = time.time()
    while True:
        update_particles = False
        if  bit > 0 and time.time() - last > fps_correspond_time:
            scene.update()
            last = time.time()
            bit -= 1
            update_particles = True
            rd.update(update_particles=update_particles)
            rd.render()
        # else:
        #     rd.update(update_particles=False)
        #     rd.render()
