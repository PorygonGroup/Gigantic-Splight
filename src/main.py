import time

import taichi as ti

from scene import Scene
from pbf3d import ParticleSystem, Simulator
import pbf3d
from renderer import Renderer

# ti.init(arch=ti.gpu)

arch = ti.cuda if ti._lib.core.with_cuda() else ti.vulkan
ti.init(arch=arch)

screen_res = (1920, 1080)
radius = 0.1
class Object(object):
    pass
def getBox(ls,h):
    obj = Object()
    obj.n0 = ls[0]
    obj.n1 = ls[1]
    obj.n2 = ls[2]
    obj.n3 = ls[3]
    obj.h = h
    return obj

if __name__ == '__main__':
    box = getBox([(0.4,0.7),(0.7,0.6),(0.6,0.3),(0.3,0.4)],0.3)
    scene = Scene()
    ps = ParticleSystem(pbf3d.particle_num, radius, scene)
    sim = Simulator(ps)
    rd = Renderer(ps, scene)
    rd.addBox(box)
    # bit = 10000
    # last = time.time()
    while True:
        scene.update()
        # if bit>0 and time.time()-last>1:
            # last = time.time()
        sim.step()
            # bit -=1

        rd.render()
