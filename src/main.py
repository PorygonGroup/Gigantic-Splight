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

if __name__ == '__main__':
    scene = Scene()
    ps = ParticleSystem(pbf3d.particle_num, pbf3d.neighbor_radius, scene)
    sim = Simulator(ps)
    rd = Renderer(ps, scene)
    bit = 10000
    last = time.time()
    while True:
        scene.update()
        if bit>0 and time.time()-last>3:
            last = time.time()
            sim.step()
            bit -=1

        rd.render()
